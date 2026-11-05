#!/usr/bin/env python
#Upload a batch's finished records to S3 while the batch renders, and free the local disk
#when it runs low. run_batch_from_config.py switches this on when it is given --s3-prefix.
#
#WHY THE aws CLI AND NOT boto3. .venv310 pins requests 2.21.0 / urllib3 1.24.3 for the
#upstream toolkit, and botocore needs urllib3 >= 1.25.4. The CLI v2 is a self-contained
#binary, and every call re-reads the credential cache, so an `aws login` typed in another
#terminal halfway through a batch takes effect on the next call without a restart.
#
#THE ORDER THAT SURVIVES A CRASH. A record moves through
#    rendered -> manifest line (fsync) -> uploaded -> verified by listing -> ledger line (fsync)
#and only a record with a ledger line is ever deleted locally. A crash anywhere leaves either
#a record that uploads again (harmless: sync skips what is already there) or a ledger record
#whose remaining local files are deleted later. Never a record that is gone locally and
#missing from the ledger, which a resume would render again.
#
#ONE WRITER PER FILE. The storage thread alone writes the ledger; the main thread alone writes
#the manifest. The thread draws no random numbers and touches neither matplotlib nor PIL: with
#max_workers 1 the render runs in this same process, and a draw from a global RNG here would
#shift the stream of the record being rendered and break its regeneration.
import os, re, sys, glob, json, time, shutil, threading, subprocess, traceback, datetime
from tqdm import tqdm

RUNNER_DIR = '_runner'
TARGET_FILE = 'target.json'
LEDGER_FILE = 'ledger.jsonl'
MANIFEST_FILE = 'manifest.jsonl'
PROTECTED_FILE = 'protected.txt'
EXPIRED_MARKER = 'AWS_SESSION_EXPIRED'
STAGE_DIR = 'stage'
MIRROR_DIR = '.mirror'

#TEST HOOK ONLY. When set to the path of a file holding a number, free_gb() reads the free
#space from that file on every call instead of asking the filesystem, so the low-disk and
#hard-floor paths can be exercised without filling a disk.
FREE_GB_FILE_ENV = 'ECG_RUNNER_FREE_GB_FILE'

UPLOAD_DEFAULTS = {
    'batch_size': 100,
    'min_free_gb': 40,
    'target_free_gb': 60,
    'hard_floor_gb': 15,
    'retry_minutes': 10,
    'timeout_minutes': 120,
    'aws_profile': 'default',
    'aws_binary': None,
    'protect_backlog': True,
}

#What the CLI prints when the `aws login` session is over. Verified 2026-11-04 with CLI
#2.36.25: "aws: [ERROR]: Your session has expired. Please reauthenticate using 'aws login'."
#The other spellings are what S3 itself answers to an expired token mid-transfer.
EXPIRED_PATTERN = re.compile(
    r'session has expired|reauthenticate|aws login|ExpiredToken|token has expired',
    re.IGNORECASE)

#A record's files are <name>-<frame>.<ext> for each frame and <name>.<ext> for the WFDB copy.
#Record names carry no dot, so the first dot ends the name and its frame suffix.
RECORD_FILE = re.compile(r'^(?P<name>[^.]+?)(?:-(?P<frame>\d+))?\.(?P<ext>.+)$')

#Annotation keys left out of the manifest: per-sample and per-node geometry, hundreds of KB a
#page, which the uploaded JSON keeps anyway. Any other value larger than the bound is left out
#too and named in dropped_keys, so a future heavy key cannot grow the manifest unnoticed.
HEAVY_KEYS = ('leads', 'gridpoints', 'gridpoints_mask')
MAX_APPLIED_VALUE_BYTES = 8192


def log(message):
    """One timestamped line in the run log, written around the progress bar.

    Under nohup the bar's carriage-return updates and these lines share one file, so a line
    starts after a newline while a bar is open - otherwise it is glued to the bar's last
    update and `grep '^\\['` misses it.
    """
    lead = '\n' if getattr(tqdm, '_instances', None) else ''
    tqdm.write('%s[%s] %s' % (lead, time.strftime('%Y-%m-%d %H:%M:%S'), message),
               file=sys.stdout)
    sys.stdout.flush()


def banner(message):
    rule = '!' * 78
    return '\n'.join((rule, message, rule))


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def validate_upload(block, config_path):
    """The upload: block of a batch YAML, with every key given its default."""
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise SystemExit("%s: upload: must be a mapping of settings, e.g. {batch_size: 100}"
                         % config_path)
    unknown = sorted(set(block) - set(UPLOAD_DEFAULTS))
    if unknown:
        raise SystemExit("%s: upload: unknown key(s): %s. Known: %s. The S3 prefix is not one "
                         "of them: it is passed on the command line as --s3-prefix."
                         % (config_path, ', '.join(unknown), ', '.join(sorted(UPLOAD_DEFAULTS))))
    settings = dict(UPLOAD_DEFAULTS)
    settings.update(block)

    def number(key, minimum=0):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= minimum:
            raise SystemExit("%s: upload: %s is %r and must be a number above %s"
                             % (config_path, key, value, minimum))

    batch_size = settings['batch_size']
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise SystemExit("%s: upload: batch_size is %r and must be a whole number of at least 1. "
                         "It is how many records one upload sends." % (config_path, batch_size))
    for key in ('min_free_gb', 'target_free_gb', 'hard_floor_gb', 'retry_minutes',
                'timeout_minutes'):
        number(key)
    if not settings['hard_floor_gb'] < settings['min_free_gb'] < settings['target_free_gb']:
        raise SystemExit(
            "%s: upload: needs hard_floor_gb < min_free_gb < target_free_gb, got %s, %s, %s. "
            "Below min_free_gb uploaded local copies are deleted until target_free_gb is free; "
            "below hard_floor_gb rendering pauses." % (
                config_path, settings['hard_floor_gb'], settings['min_free_gb'],
                settings['target_free_gb']))
    if not isinstance(settings['aws_profile'], str) or not settings['aws_profile']:
        raise SystemExit("%s: upload: aws_profile must be a profile name" % config_path)
    if settings['aws_binary'] is not None and (
            not isinstance(settings['aws_binary'], str) or not settings['aws_binary']):
        raise SystemExit("%s: upload: aws_binary must be a path to the aws executable"
                         % config_path)
    if not isinstance(settings['protect_backlog'], bool):
        raise SystemExit("%s: upload: protect_backlog must be true or false" % config_path)
    return settings


def parse_s3_uri(uri):
    """s3://bucket/prefix -> (bucket, prefix), the prefix without its slashes at either end."""
    match = re.match(r'^s3://([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])/(.*)$', uri or '')
    prefix = match.group(2).strip('/') if match else ''
    if not prefix:
        raise ValueError(
            "--s3-prefix must be s3://<bucket>/<prefix> with a non-empty prefix, got %r. "
            "The records land flat under the prefix, so the bucket root is refused." % uri)
    return match.group(1), prefix


def s3_name(filename):
    """The name a local file takes on S3: the digitizer's `make upload-dataset` rule.

    Its awk applies gsub(/_lr-0/, "") and then gsub(/_lr/, ""), so 00001_lr-0.png becomes
    00001.png and 00001_lr.dat becomes 00001.dat. That flat layout is what the digitizer's
    deploy/sagemaker/build_cache_s3.py reads, so it is mirrored exactly.
    """
    return filename.replace('_lr-0', '').replace('_lr', '')


def index_output(output_root):
    """Every record file directly in the output directory, grouped by record name."""
    index = {}
    for entry in os.scandir(output_root):
        if not entry.is_file():
            continue
        match = RECORD_FILE.match(entry.name)
        if match:
            index.setdefault(match.group('name'), []).append(entry.name)
    for files in index.values():
        files.sort()
    return index


def record_files(output_root, name):
    """The files one record has in the output directory, by name."""
    found = []
    for path in glob.glob(glob.escape(os.path.join(output_root, name)) + '[-.]*'):
        base = os.path.basename(path)
        match = RECORD_FILE.match(base)
        if match and match.group('name') == name and os.path.isfile(path):
            found.append(base)
    return sorted(found)


def frame_files(files, name):
    """The frame PNGs among a record's files, sorted by frame number."""
    frames = []
    for base in files:
        match = RECORD_FILE.match(base)
        if match and match.group('name') == name and match.group('frame') is not None \
                and match.group('ext') == 'png':
            frames.append((int(match.group('frame')), base))
    return [base for _, base in sorted(frames)]


def free_gb(path):
    override = os.environ.get(FREE_GB_FILE_ENV)
    if override:
        try:
            with open(override) as handle:
                return float(handle.read().strip())
        except (OSError, ValueError):
            pass
    return shutil.disk_usage(path).free / 2 ** 30


def _jsonable(value):
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def write_atomic(path, text):
    """Replace a file in one step: a crash leaves the old content or the new, never half."""
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(os.path.dirname(path) or '.', os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class JsonlFile:
    """An append-only file of JSON lines, fsynced on every append."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def repair(self):
        """Cut a last line a crash left without its newline, so the next append starts clean."""
        if not os.path.exists(self.path):
            return
        with open(self.path, 'rb+') as handle:
            data = handle.read()
            if not data or data.endswith(b'\n'):
                return
            cut = data.rfind(b'\n') + 1
            handle.truncate(cut)
        log("upload: %s ended in a line cut off mid-write; the line was dropped" % self.path)

    def entries(self):
        self.repair()
        if not os.path.exists(self.path):
            return
        with open(self.path) as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    log("upload: %s line %d is not JSON and was ignored" % (self.path, number))

    def append(self, entries):
        text = ''.join(json.dumps(entry, sort_keys=True, default=_jsonable) + '\n'
                       for entry in entries)
        with self.lock:
            with open(self.path, 'a') as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())

    def snapshot(self, destination):
        """A copy taken under the append lock, so it never holds half a line."""
        with self.lock:
            if not os.path.exists(self.path):
                return False
            shutil.copyfile(self.path, destination)
            return True


class AwsResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout or ''
        self.stderr = stderr or ''
        self.ok = returncode == 0
        self.expired = not self.ok and bool(
            EXPIRED_PATTERN.search(self.stderr) or EXPIRED_PATTERN.search(self.stdout))

    def summary(self):
        lines = [line.strip() for line in (self.stderr or self.stdout).splitlines() if line.strip()]
        return (' | '.join(lines[-3:]) or 'no output')[:500]


class Aws:
    """The aws CLI v2, called by absolute path with the profile of the batch."""

    def __init__(self, binary, profile, timeout_minutes):
        self.binary = binary
        self.profile = profile
        self.timeout = timeout_minutes * 60

    def run(self, arguments, timeout=None):
        command = [self.binary, '--profile', self.profile, '--no-cli-pager'] + list(arguments)
        #AWS_PAGER empty as well: CLI v2 pipes s3api output through less when it sees a tty.
        environment = dict(os.environ, AWS_PAGER='')
        try:
            done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, env=environment, timeout=timeout or self.timeout)
        except subprocess.TimeoutExpired:
            return AwsResult(-1, '', 'timed out after %.0f min' % ((timeout or self.timeout) / 60))
        except OSError as error:
            return AwsResult(-1, '', 'could not run %s: %s' % (self.binary, error))
        return AwsResult(done.returncode, done.stdout, done.stderr)

    def list_prefix(self, bucket, prefix):
        """key -> size of every object under the prefix, or (None, result) on failure."""
        result = self.run(['s3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix,
                           '--output', 'json', '--query', 'Contents[].[Key,Size]'])
        if not result.ok:
            return None, result
        try:
            rows = json.loads(result.stdout) if result.stdout.strip() else None
        except ValueError:
            return None, AwsResult(-1, '', 'the listing of s3://%s/%s is not JSON' % (bucket, prefix))
        return {key: size for key, size in rows or []}, result


def resolve_aws_binary(configured):
    """An absolute path to the CLI, fixed at startup.

    A batch left running under nohup or tmux can outlive the PATH it started with, so the
    binary is never looked up again after this.
    """
    candidates = [os.path.expanduser(configured)] if configured else \
        [shutil.which('aws'), os.path.expanduser('~/.local/bin/aws')]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
    raise SystemExit("upload: no executable aws CLI at %s. Install the AWS CLI v2 or set "
                     "upload: aws_binary." % (', '.join(c for c in candidates if c) or 'PATH'))


def plan_adoption(listing, prefix, index, sizes, finished):
    """What a first start makes of what the prefix already holds. Pure: reads nothing.

    Returns (adopted, divergent, runner_keys):
      adopted      finished local records whose every file is on S3 with the same size;
      divergent    (key, s3 size, local size or None) for each object that is NOT a copy of a
                   local file - any of them means the prefix holds something else;
      runner_keys  objects under <prefix>/_runner/, which mean an upload run already started
                   into this prefix from somewhere.
    """
    runner_root = '%s/%s/' % (prefix, RUNNER_DIR)
    local_by_s3 = {s3_name(base): base for files in index.values() for base in files}
    runner_keys, divergent = [], []
    for key, size in sorted(listing.items()):
        if key.startswith(runner_root):
            runner_keys.append(key)
            continue
        base = local_by_s3.get(key[len(prefix) + 1:])
        if base is None:
            divergent.append((key, size, None))
        elif sizes[base] != size:
            divergent.append((key, size, sizes[base]))
    adopted = sorted(
        name for name in finished
        if all(listing.get('%s/%s' % (prefix, s3_name(base))) == sizes[base]
               for base in index[name]))
    return adopted, divergent, runner_keys


class Storage:
    """The storage thread: uploads finished records in lots and deletes uploaded copies."""

    def __init__(self, settings, bucket, prefix, output_root, aws):
        self.settings = settings
        self.bucket = bucket
        self.prefix = prefix
        self.uri_root = 's3://%s/%s' % (bucket, prefix)
        self.output_root = output_root
        self.aws = aws
        self.runner_dir = os.path.join(output_root, RUNNER_DIR)
        self.ledger_file = JsonlFile(os.path.join(self.runner_dir, LEDGER_FILE))
        self.manifest = JsonlFile(os.path.join(self.runner_dir, MANIFEST_FILE))
        self.ledger = {}        #name -> ledger entry
        self.protected = set()  #names never deleted locally (the backlog of the first start)
        self.pending = {}       #name -> its local files: finished, not uploaded yet
        self.local = set()      #ledger names that may still have local files
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.flushing = False
        self.paused = False
        self.next_attempt = 0.0
        self.expired_since = None
        self.last_banner = 0.0
        self.last_stuck_note = 0.0
        self.session = time.strftime('%Y%m%dT%H%M%S')
        self.lot_number = 0
        self.thread = None

    #----------------------------------------------------------------- the main thread's side

    def key(self, base):
        return '%s/%s' % (self.prefix, s3_name(base))

    def uri(self, base):
        return '%s/%s' % (self.uri_root, s3_name(base))

    def add_finished(self, name, files):
        with self.lock:
            self.pending[name] = list(files)
        self.wakeup.set()

    def start(self):
        self.thread = threading.Thread(target=self._run, name='s3-upload', daemon=True)
        self.thread.start()

    def describe(self):
        with self.lock:
            waiting = len(self.pending)
        deletable = len([name for name in self.local if name not in self.protected])
        return ('%d waiting for upload, %d in the ledger, %d uploaded copies deletable, '
                '%d protected%s' % (waiting, len(self.ledger), deletable, len(self.protected),
                                    '; AWS SESSION EXPIRED' if self.expired_since else ''))

    def wait_for_space(self):
        """Pause the render, without aborting it, while the disk is below the hard floor."""
        free = free_gb(self.output_root)
        if free >= self.settings['hard_floor_gb']:
            return
        self.paused = True
        self.wakeup.set()
        log(banner("DISK BELOW THE HARD FLOOR: %.1f GB free, hard_floor_gb is %s. Rendering is "
                   "PAUSED, not aborted, until %s GB are free; the storage thread keeps\nuploading "
                   "and deleting uploaded copies. %s" % (
                       free, self.settings['hard_floor_gb'], self.settings['min_free_gb'],
                       self.describe())))
        last = time.time()
        while free_gb(self.output_root) < self.settings['min_free_gb']:
            time.sleep(15)
            if time.time() - last >= 600:
                log("upload: still paused at %.1f GB free - %s" % (free_gb(self.output_root),
                                                                    self.describe()))
                if not self.thread.is_alive():
                    log("upload: THE STORAGE THREAD IS NOT RUNNING; free space by hand")
                last = time.time()
        self.paused = False
        log("upload: %.1f GB free, rendering resumed" % free_gb(self.output_root))

    def finish(self):
        """Upload whatever is left, the last partial lot included, and wait for it."""
        self.flushing = True
        self.wakeup.set()
        last = time.time()
        while self.thread.is_alive():
            self.thread.join(30)
            if self.thread.is_alive() and time.time() - last >= 600:
                log("upload: waiting for the last records to reach S3 - %s. Ctrl-C is safe: "
                    "the next run uploads them." % self.describe())
                last = time.time()

    #------------------------------------------------------------------ the thread's own side

    def _run(self):
        self._mirror()
        while True:
            try:
                state = self._step()
            except Exception:
                log("upload: unexpected error in the storage thread, retrying in %s min:\n%s"
                    % (self.settings['retry_minutes'], traceback.format_exc()))
                self.next_attempt = time.time() + self.settings['retry_minutes'] * 60
                state = 'idle'
            if state == 'done':
                return
            if state == 'more':
                continue
            self.wakeup.wait(30)
            self.wakeup.clear()

    def _step(self):
        with self.lock:
            waiting = len(self.pending)
        low = free_gb(self.output_root) < self.settings['min_free_gb']
        urgent = self.flushing or self.paused or low
        if waiting and time.time() >= self.next_attempt and (
                waiting >= self.settings['batch_size'] or urgent):
            if self._upload_lot():
                return 'more'
            self.next_attempt = time.time() + self.settings['retry_minutes'] * 60
        if low:
            self._free_space()
        if self.flushing:
            with self.lock:
                if not self.pending:
                    return 'done'
        return 'idle'

    def _free_space(self):
        """Delete uploaded, unprotected local copies, oldest name first, up to target_free_gb."""
        target = self.settings['target_free_gb']
        deleted, freed = 0, 0
        for name in sorted(self.local):
            if free_gb(self.output_root) >= target:
                break
            if name in self.protected:
                continue
            for base in self.ledger[name]['files']:
                path = os.path.join(self.output_root, base)
                try:
                    size = os.path.getsize(path)
                    os.remove(path)
                    freed += size
                except FileNotFoundError:
                    pass
            self.local.discard(name)
            deleted += 1
        free = free_gb(self.output_root)
        if deleted:
            log("upload: deleted the local copy of %d uploaded record(s), %.2f GB; %.1f GB free"
                % (deleted, freed / 2 ** 30, free))
        if free < target and time.time() - self.last_stuck_note >= 3600:
            if not [name for name in self.local if name not in self.protected]:
                log("upload: %.1f GB free, below target_free_gb %s, and no uploaded copy is left "
                    "to delete - %s" % (free, target, self.describe()))
                self.last_stuck_note = time.time()

    def _upload_lot(self):
        with self.lock:
            names = sorted(self.pending)[:self.settings['batch_size']]
            lot = [(name, list(self.pending[name])) for name in names]
        self.lot_number += 1
        lot_id = '%s/%d' % (self.session, self.lot_number)
        stage = os.path.join(self.runner_dir, STAGE_DIR, 'lot-%d' % self.lot_number)
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        expected, ready = {}, []
        for name, files in lot:
            missing = [base for base in files
                       if not os.path.isfile(os.path.join(self.output_root, base))]
            if missing:
                log("upload: %s lost %s since it was rendered; dropped from the upload, the next "
                    "run renders it again" % (name, ', '.join(missing)))
                with self.lock:
                    self.pending.pop(name, None)
                continue
            for base in files:
                link = os.path.join(stage, s3_name(base))
                if os.path.lexists(link):
                    raise RuntimeError("two local files map to %s on S3" % s3_name(base))
                path = os.path.join(self.output_root, base)
                os.symlink(path, link)
                expected[self.key(base)] = os.path.getsize(path)
            ready.append((name, files))
        if not ready:
            shutil.rmtree(stage, ignore_errors=True)
            return True

        started = time.time()
        #sync, not cp: a retry after a partial failure skips the objects already there with
        #the same size, and the CLI follows the staging symlinks by default.
        result = self.aws.run(['s3', 'sync', stage + '/', self.uri_root + '/',
                               '--only-show-errors', '--no-progress'])
        if not result.ok:
            self._failed('upload of lot %s (%d records)' % (lot_id, len(ready)), result)
            return False
        elapsed = time.time() - started
        #An exit 0 is not the proof: the listing is. Every object of the lot must be there with
        #the size of its local file before a single ledger line is written.
        listing, result = self.aws.list_prefix(self.bucket, self.prefix + '/')
        if listing is None:
            self._failed('verification listing of lot %s' % lot_id, result)
            return False
        wrong = sorted(key for key, size in expected.items() if listing.get(key) != size)
        if wrong:
            log("upload: lot %s did not verify - %d object(s) missing or of another size on S3, "
                "e.g. %s. Nothing recorded, nothing deleted; retrying."
                % (lot_id, len(wrong), ', '.join(wrong[:3])))
            return False

        stamp = now_iso()
        entries = [{
            'record': name,
            'source': 'upload',
            'frames': len(frame_files(files, name)),
            'files': files,
            's3': [self.uri(base) for base in files],
            'bytes': sum(expected[self.key(base)] for base in files),
            'lot': lot_id,
            'at': stamp,
        } for name, files in ready]
        self.ledger_file.append(entries)
        with self.lock:
            for name, _ in ready:
                self.pending.pop(name, None)
        for entry in entries:
            self.ledger[entry['record']] = entry
            self.local.add(entry['record'])
        shutil.rmtree(stage, ignore_errors=True)
        self._session_back()

        total = sum(expected.values())
        log("upload: lot %s - %d record(s), %.2f GB in %.0f s (%.1f MB/s); %d in the ledger; "
            "%.1f GB free" % (lot_id, len(ready), total / 2 ** 30, elapsed,
                              total / 2 ** 20 / max(elapsed, 1e-6), len(self.ledger),
                              free_gb(self.output_root)))
        self._mirror()
        return True

    def _mirror(self):
        """Copy _runner/ - ledger, manifest, target, protected list, configs - next to the data."""
        mirror = os.path.join(self.runner_dir, MIRROR_DIR)
        os.makedirs(mirror, exist_ok=True)
        destination = '%s/%s/' % (self.uri_root, RUNNER_DIR)
        result = self.aws.run(['s3', 'sync', self.runner_dir + '/', destination,
                               '--only-show-errors', '--no-progress',
                               '--exclude', STAGE_DIR + '/*', '--exclude', MIRROR_DIR + '/*',
                               '--exclude', MANIFEST_FILE, '--exclude', EXPIRED_MARKER,
                               '--exclude', '*.tmp'])
        snapshot = os.path.join(mirror, MANIFEST_FILE)
        if result.ok and self.manifest.snapshot(snapshot):
            result = self.aws.run(['s3', 'cp', snapshot, destination + MANIFEST_FILE,
                                   '--only-show-errors', '--no-progress'])
        if not result.ok:
            if result.expired:
                self._failed('copy of _runner/ to S3', result)
            else:
                log("upload: copy of _runner/ to S3 failed (%s); retried after the next lot"
                    % result.summary())
        return result.ok

    def _failed(self, what, result):
        if result.expired:
            now = time.time()
            first = self.expired_since is None
            if first:
                self.expired_since = now
                with open(os.path.join(self.runner_dir, EXPIRED_MARKER), 'w') as handle:
                    handle.write('%s\n' % now_iso())
            if first or now - self.last_banner >= 3600:
                log(banner("AWS SESSION EXPIRED - run `aws login` (profile %s).\nUpload is "
                           "paused and rendering continues. Nothing that is not on S3 is "
                           "deleted; retrying every %s min.\n%s" % (
                               self.settings['aws_profile'], self.settings['retry_minutes'],
                               self.describe())))
                self.last_banner = now
            else:
                log("upload: AWS session still expired (since %s); retrying in %s min" % (
                    time.strftime('%H:%M', time.localtime(self.expired_since)),
                    self.settings['retry_minutes']))
        else:
            log("upload: %s failed (exit %s): %s - retrying in %s min" % (
                what, result.returncode, result.summary(), self.settings['retry_minutes']))

    def _session_back(self):
        if self.expired_since is not None:
            log("upload: the AWS session is valid again; upload resumed")
            self.expired_since = None
            try:
                os.remove(os.path.join(self.runner_dir, EXPIRED_MARKER))
            except FileNotFoundError:
                pass


def open_storage(settings, s3_uri, output_root, finished, entry_for):
    """Prepare the storage thread for a run: first start (adoption) or resume.

    finished(name, backfill) returns the record's annotations, one dict per frame, when it is
    finished on disk, else None; backfill=True asks for the stricter proof a record WITHOUT a
    manifest line needs. entry_for(name, source, annotations, files) builds a manifest line.
    Returns (storage, names finished on disk and waiting for upload).
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    aws = Aws(resolve_aws_binary(settings['aws_binary']), settings['aws_profile'],
              settings['timeout_minutes'])
    storage = Storage(settings, bucket, prefix, output_root, aws)
    os.makedirs(storage.runner_dir, exist_ok=True)
    shutil.rmtree(os.path.join(storage.runner_dir, STAGE_DIR), ignore_errors=True)

    index = index_output(output_root)
    manifest_files = {}
    for entry in storage.manifest.entries():
        manifest_files[entry['record']] = entry.get('files') or []

    target_path = os.path.join(storage.runner_dir, TARGET_FILE)
    finished_map = {}
    if os.path.exists(target_path):
        with open(target_path) as handle:
            target = json.load(handle)
        if target.get('s3_prefix') != storage.uri_root:
            raise SystemExit(
                "upload: %s uploads to %s, not %s. A directory goes to one prefix only - pass "
                "the same --s3-prefix, or use another output_directory."
                % (output_root, target.get('s3_prefix'), storage.uri_root))
        #A resume renders whatever the session: the check below only warns, after the ledger
        #is loaded so the warning can say what is waiting.
        check_session = True
    else:
        check_session = False
        result = aws.run(['sts', 'get-caller-identity'], timeout=120)
        if not result.ok:
            if result.expired:
                raise SystemExit(
                    "upload: the AWS session is expired. The first start of an upload into %s "
                    "checks what %s already holds and needs S3 for that: run `aws login` and "
                    "start again." % (output_root, storage.uri_root))
            raise SystemExit("upload: aws sts get-caller-identity failed: %s" % result.summary())
        listing, result = aws.list_prefix(bucket, prefix + '/')
        if listing is None:
            raise SystemExit("upload: cannot list %s: %s" % (storage.uri_root, result.summary()))
        sizes = {base: os.path.getsize(os.path.join(output_root, base))
                 for files in index.values() for base in files}
        log("upload: first start into %s - checking %d local record(s) against %d object(s)"
            % (storage.uri_root, len(index), len(listing)))
        #A yes or no per record, never the annotations themselves: each one holds every
        #plotted sample of the page, several MB as Python objects, and the backlog this first
        #start reads is thousands of records. The backfill below reads them again one at a time.
        finished_map = {name: finished(name, True) is not None for name in sorted(index)}
        finished_set = {name for name, done in finished_map.items() if done}
        adopted, divergent, runner_keys = plan_adoption(listing, prefix, index, sizes,
                                                        finished_set)
        if runner_keys:
            raise SystemExit(
                "upload: %s already holds %s/ (%d objects) but %s has no %s. This dataset was "
                "started from another directory, or its _runner/ was lost. To continue it, "
                "restore that directory first:\n    aws s3 cp --recursive %s/%s/ %s/\nor "
                "choose another prefix." % (
                    storage.uri_root, RUNNER_DIR, len(runner_keys), output_root, TARGET_FILE,
                    storage.uri_root, RUNNER_DIR, storage.runner_dir))
        if divergent:
            examples = '\n'.join('    %s: %s bytes on S3, %s' % (
                key, size, 'no local file' if local is None else '%s bytes locally' % local)
                for key, size, local in divergent[:10])
            raise SystemExit(
                "upload: %s holds %d object(s) that are not a copy of a file in %s, e.g.\n%s\n"
                "Uploading would overwrite another dataset. Choose a prefix that is empty or "
                "holds a copy of this directory." % (
                    storage.uri_root, len(divergent), output_root, examples))

        protected = sorted(finished_set) if settings['protect_backlog'] else []
        stamp = now_iso()
        adopted_entries = [{
            'record': name,
            'source': 'adopted',
            'frames': len(frame_files(index[name], name)),
            'files': index[name],
            's3': [storage.uri(base) for base in index[name]],
            'bytes': sum(sizes[base] for base in index[name]),
            'lot': None,
            'at': stamp,
        } for name in adopted]
        #protected.txt, then the ledger, then target.json, each replaced in one step: the
        #target is written last because its presence is what makes the next start a resume.
        write_atomic(os.path.join(storage.runner_dir, PROTECTED_FILE),
                     ''.join(name + '\n' for name in protected))
        write_atomic(storage.ledger_file.path,
                     ''.join(json.dumps(entry, sort_keys=True) + '\n'
                             for entry in adopted_entries))
        write_atomic(target_path, json.dumps({
            's3_prefix': storage.uri_root,
            'bucket': bucket,
            'prefix': prefix,
            'output_directory': output_root,
            'created': stamp,
            'adopted': len(adopted),
            'protected': len(protected),
        }, indent=2, sort_keys=True) + '\n')
        log("upload: adopted %d record(s) already on S3 with the same sizes; %d record(s) "
            "protected from deletion" % (len(adopted), len(protected)))

    for entry in storage.ledger_file.entries():
        storage.ledger[entry['record']] = entry
    protected_path = os.path.join(storage.runner_dir, PROTECTED_FILE)
    if os.path.exists(protected_path):
        with open(protected_path) as handle:
            storage.protected = {line.strip() for line in handle if line.strip()}
    storage.local = {name for name in storage.ledger if name in index}

    #Backfill: a record finished on disk with no manifest line - the backlog of the first
    #start, or a record a kill caught between its render and its line - gets one now, built
    #from its annotation. It must pass the stricter proof, because nothing this process saw
    #says its render returned.
    backfill = []
    for name in sorted(index):
        if name in manifest_files:
            continue
        if name in finished_map and not finished_map[name]:
            continue
        annotations = finished(name, True)
        if annotations is None:
            continue
        backfill.append(entry_for(name, 'backfill', annotations, index[name]))
        manifest_files[name] = index[name]
        if len(backfill) >= 500:
            storage.manifest.append(backfill)
            backfill = []
    if backfill:
        storage.manifest.append(backfill)

    waiting = set()
    for name, files in manifest_files.items():
        if name in storage.ledger or name not in index:
            continue
        on_disk = set(index[name])
        if not files or not all(base in on_disk for base in files):
            continue
        done = finished_map[name] if name in finished_map else finished(name, False) is not None
        if not done:
            continue
        storage.pending[name] = list(files)
        waiting.add(name)
    if check_session:
        result = aws.run(['sts', 'get-caller-identity'], timeout=120)
        if not result.ok:
            storage._failed('session check', result)
    log("upload: %s - %s" % (storage.uri_root, storage.describe()))
    return storage, waiting
