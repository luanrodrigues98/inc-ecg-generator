import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import LineCollection
from matplotlib.transforms import Bbox
from scipy.ndimage import gaussian_filter1d
from TemplateFiles.generate_template import generate_template
from math import ceil 
from PIL import Image
import csv

standard_values = {'y_grid_size' : 0.5,
                   'x_grid_size' : 0.2,
                   'y_grid_inch' : 5/25.4,
                   'x_grid_inch' : 5/25.4,
                   'grid_line_width' : 0.5,
                   'lead_name_offset' : 0.5,
                   'lead_fontsize' : 11,
                   'x_gap' : 1,
                   'y_gap' : 0.5,
                   'display_factor' : 1,
                   'line_width': 0.75,
                   'row_height' : 8,
                   'dc_offset_length' : 0.2,
                   'lead_length' : 3,
                   'V1_length' : 12,
                   'width' : 11,
                   'height' : 8.5,
                   'trace_thickness_noise_length_mm' : 5.0,
                   'trace_dropout_full_gap_share' : 0.3,
                   'trace_dropout_min_opacity' : 0.3,
                   'trace_dropout_max_opacity' : 0.7,
                   'trace_dropout_max_length_factor' : 5.0
                   }

standard_major_colors = {'colour1' : (0.4274,0.196,0.1843), #brown
                          'colour2' : (1,0.796,0.866), #pink
                          'colour3' : (0.0,0.0, 0.4), #blue
                          'colour4' : (0,0.3,0.0), #green
                          'colour5' : (1,0,0), #red
                          'colour6' : (0.4,0.4,0.4) #black (grey major rule, as in the bw path; sits clear of the near-black trace)
    }


standard_minor_colors = {'colour1' : (0.5882,0.4196,0.3960),
                         'colour2' : (0.996,0.9294,0.9725),
                         'colour3' : (0.0,0, 0.7),
                         'colour4' : (0,0.8,0.3),
                         'colour5' : (0.996,0.8745,0.8588),
                         'colour6' : (0.75,0.75,0.75) #black (light-grey minor rule, as in the bw path)
    }

papersize_values = {'A0' : (33.1,46.8),
                    'A1' : (33.1,23.39),
                    'A2' : (16.54,23.39),
                    'A3' : (11.69,16.54),
                    'A4' : (8.27,11.69),
                    'letter' : (8.5,11)
                    }

#The lattice spacing for the --store_gridpoints ground truth: one node per major (5mm)
#grid square, matching what a UNet-based rectifier (e.g. hengck23's Kaggle notebook) is
#trained to detect. Fixed, not a config.yaml knob - x_grid_size/y_grid_size already
#denote exactly this 5mm step (see standard_values above), this constant only documents
#that fact for readers of the emitted JSON.
GRIDPOINT_MM_PER_STEP = 5.0


def inches_to_dots(value,resolution):
    return (value * resolution)

def mm_to_points(value_mm):
    #Convert a paper length in millimetres to matplotlib points (1/72 inch).
    #Points are a physical unit, so a width derived this way is invariant to the
    #render resolution.
    return value_mm * 72.0 / 25.4

def variable_trace_widths(num_points,base_width_points,jitter,samples_per_mm,correlation_length_mm,rng):
    #Per-segment stroke widths for a trace of num_points points.
    #The width is modulated with smooth 1D noise to imitate the variable pressure of a
    #stylus; a perfectly constant stroke width is the strongest giveaway of a synthetic
    #ECG. Returns num_points - 1 widths, one per segment, bounded to +/- jitter.
    num_segments = max(num_points - 1, 1)
    sigma = max(correlation_length_mm * samples_per_mm, 1.0)
    noise = gaussian_filter1d(rng.standard_normal(num_segments), sigma=sigma, mode='reflect')
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak
    return base_width_points * (1.0 + jitter * noise)

def trace_arc_length_mm(x_vals,y_vals):
    #Cumulative arc length of the trace, in millimetres of paper.
    #The two axes carry different physical scales - x runs at 25 mm/s and y at 10 mm/mV -
    #so a step on each has to be converted before the two are combined. Non-finite
    #samples come from --mask_unplotted_samples, and a single NaN would poison the
    #cumulative sum and silently suppress every dropout downstream.
    mm_per_x = standard_values['x_grid_inch']*25.4/standard_values['x_grid_size']
    mm_per_y = standard_values['y_grid_inch']*25.4/standard_values['y_grid_size']
    dx = np.nan_to_num(np.diff(np.asarray(x_vals,dtype=float)))*mm_per_x
    dy = np.nan_to_num(np.diff(np.asarray(y_vals,dtype=float)))*mm_per_y
    return np.concatenate(([0.0],np.cumsum(np.hypot(dx,dy))))

def trace_dropout_opacities(x_vals,y_vals,dropout_rate,dropout_length_mm,rng):
    #Per-segment opacity of the ecg trace: 1.0 where the ink is laid down normally, 0.0
    #inside a full gap and 0.3-0.7 where the ink only fades. Imitates a stylus losing
    #contact or ink failing.
    #The dropouts are placed along the ARC LENGTH of the trace rather than along x: on a
    #near vertical R wave the trace covers a lot of arc in very little x, and a dropout
    #has to be able to land in the middle of the upstroke.
    arc = trace_arc_length_mm(x_vals,y_vals)
    num_segments = max(len(x_vals) - 1,1)
    opacities = np.ones(num_segments)
    total_mm = float(arc[-1])
    if total_mm <= 0:
        return opacities
    #dropout_rate counts dropouts per cm of trace
    num_dropouts = int(rng.poisson(dropout_rate*total_mm/10.0))
    if num_dropouts == 0:
        return opacities
    starts = rng.uniform(0.0,total_mm,num_dropouts)
    #Exponential lengths rather than uniform: short gaps are far more common than long
    #ones. The clip keeps a single draw from the tail from swallowing a whole lead.
    lengths = np.clip(rng.exponential(dropout_length_mm,num_dropouts),
                      0.05,
                      dropout_length_mm*standard_values['trace_dropout_max_length_factor'])
    #Partial fading is the more common case in real thermal printing, so only a minority
    #of the dropouts take the ink away completely.
    is_full_gap = rng.random(num_dropouts) < standard_values['trace_dropout_full_gap_share']
    faded = rng.uniform(standard_values['trace_dropout_min_opacity'],
                        standard_values['trace_dropout_max_opacity'],
                        num_dropouts)
    values = np.where(is_full_gap,0.0,faded)
    #Each dropout is blended into the segments it overlaps, in proportion to how much of
    #the segment arc it covers, rather than by whole segments. Segment membership would
    #let a short dropout blank a long segment: on a steep R wave a single sample step
    #spans several mm of arc, and a 0.5 mm ink failure must not erase all of it. Blending
    #also softens the two segments at the edge of a gap.
    #The removed ink accumulates, so two dropouts landing on the same segment compound
    #towards a full gap instead of the more severe one hiding the other.
    segment_lengths = np.diff(arc)
    removed = np.zeros(num_segments)
    for start,length,value in zip(starts,lengths,values):
        lo = max(int(np.searchsorted(arc,start,side='right')) - 1,0)
        hi = min(int(np.searchsorted(arc,start + length,side='left')) + 1,num_segments)
        if hi <= lo:
            continue
        overlap = (np.minimum(arc[lo + 1:hi + 1],start + length)
                   - np.maximum(arc[lo:hi],start))
        coverage = np.clip(overlap/np.maximum(segment_lengths[lo:hi],1e-12),0.0,1.0)
        removed[lo:hi] += coverage*(1.0 - value)
    return np.clip(opacities - removed,0.0,1.0)

def trace_pieces(segments,widths,opacities,merge_opaque):
    #Group the per-segment polyline into the pieces that are actually stroked, dropping
    #the segments a full gap removed.
    #A run of faded segments is emitted as ONE polyline at the mean width of the run
    #instead of one path per segment: where two translucent strokes overlap, their round
    #caps composite to a darker value than either (two strokes at 0.5 give 0.75), which
    #would read as beading exactly inside the dropouts. Opaque strokes do not composite,
    #so they keep their own width and the existing per-segment modulation.
    paths = []
    path_widths = []
    path_opacities = []
    num_segments = len(segments)
    index = 0
    while index < num_segments:
        opacity = opacities[index]
        run = index
        while run < num_segments and opacities[run] == opacity:
            run += 1
        if opacity > 0:
            if opacity < 1.0 or merge_opaque:
                paths.append(np.concatenate([segments[index][:1],segments[index:run][:,1]]))
                path_widths.append(float(np.mean(widths[index:run])))
                path_opacities.append(float(opacity))
            else:
                for k in range(index,run):
                    paths.append(segments[k])
                    path_widths.append(float(widths[k]))
                    path_opacities.append(1.0)
        index = run
    return paths,path_widths,path_opacities

def draw_trace(ax,x_vals,y_vals,color_line,trace_style,need_bbox):
    #Draw one ecg trace and return its bounding box in display coordinates.
    #With no jitter and no dropouts the trace is a plain Line2D, reproducing the upstream
    #render exactly. Beyond that the stroke width varies along the trace and dropouts take
    #ink away, both of which require a LineCollection: a Line2D carries a single width and
    #a single opacity for the whole polyline.
    opacities = None
    if trace_style['dropout_rate'] > 0 and trace_style['dropout_length_mm'] > 0:
        opacities = trace_dropout_opacities(x_vals,y_vals,trace_style['dropout_rate'],
                                            trace_style['dropout_length_mm'],
                                            trace_style['dropout_rng'])
    if trace_style['jitter'] <= 0 and opacities is None:
        artist = ax.plot(x_vals,
                y_vals,
                linewidth=trace_style['line_width'],
                color=color_line
                )[0]
        return artist.get_window_extent() if need_bbox else None

    points = np.column_stack([x_vals, y_vals]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if trace_style['jitter'] > 0:
        widths = variable_trace_widths(len(x_vals), trace_style['line_width'], trace_style['jitter'],
                                       trace_style['samples_per_mm'], trace_style['correlation_length_mm'],
                                       trace_style['rng'])
    else:
        widths = np.full(len(segments), trace_style['line_width'])
    if opacities is None:
        opacities = np.ones(len(segments))
    paths, path_widths, path_opacities = trace_pieces(segments, widths, opacities,
                                                      trace_style['jitter'] <= 0)
    if paths:
        colors = np.tile(matplotlib.colors.to_rgba(color_line), (len(paths), 1))
        colors[:, 3] = path_opacities
        ax.add_collection(LineCollection(paths, linewidths=path_widths, colors=colors,
                                         capstyle='round', joinstyle='round', zorder=2))
    if not need_bbox:
        return None
    #Line2D.get_window_extent is the extent of the transformed data points, ignoring
    #stroke width, so the same value is reproduced here instead of read off the
    #collection. Non-finite points are skipped, matching matplotlib.
    xy = ax.transData.transform(np.column_stack([x_vals, y_vals]))
    return Bbox([[np.nanmin(xy[:, 0]), np.nanmin(xy[:, 1])],
                 [np.nanmax(xy[:, 0]), np.nanmax(xy[:, 1])]])

def build_gridpoints(ax, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size, height):
    #The --store_gridpoints ground truth: every major (5mm) grid-line intersection over
    #the whole figure rectangle, in the same pixel convention as plotted_pixels.
    #Reuses the exact tick arrays the grid itself is drawn from (np.arange(x_min,x_max,
    #x_grid_size) / the y equivalent), so the lattice always matches what is actually on
    #the page, whether or not show_grid draws it. No RNG use - ax.transData is already
    #fixed by set_xlim/set_ylim, called well before this.
    xs = np.arange(x_min, x_max, x_grid_size)
    ys = np.arange(y_min, y_max, y_grid_size)
    n_cols = len(xs)
    n_rows = len(ys)

    grid_x, grid_y = np.meshgrid(xs, ys)
    transformed = ax.transData.transform(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    px = transformed[:, 0]
    py = height - transformed[:, 1]
    #Flat, row-major over (n_rows,n_cols), [x,y] like plotted_pixels - same axis order
    #hengck23's own .npy sidecar uses, so the JSON mirror and the .npy now agree.
    gridpoints = [[round(float(x), 2), round(float(y), 2)] for y, x in zip(py, px)]

    #Empirical pixel spacing of one 5mm step, read straight off the same transform
    #rather than re-derived from x_grid_dots/y_grid_dots, so it stays correct even if
    #those are redefined elsewhere. transData is affine here (pre-crumple), so any one
    #step is representative.
    origin = ax.transData.transform((x_min, y_min))
    one_x_step = ax.transData.transform((x_min + x_grid_size, y_min))
    one_y_step = ax.transData.transform((x_min, y_min + y_grid_size))
    dx = abs(one_x_step[0] - origin[0])
    dy = abs(one_y_step[1] - origin[1])
    reference_hw = [round((n_rows - 1) * dy, 2), round((n_cols - 1) * dx, 2)]

    return gridpoints, [n_rows, n_cols], reference_hw

#Function to plot raw ecg signal
def ecg_plot(
        ecg, 
        configs,
        sample_rate, 
        columns,
        rec_file_name,
        output_dir,
        resolution,
        pad_inches,
        lead_index,
        full_mode,
        store_text_bbox,
        full_header_file,
        units          = '',
        papersize      = '',
        x_gap          = standard_values['x_gap'],
        y_gap          = standard_values['y_gap'],
        display_factor = standard_values['display_factor'],
        line_width     = standard_values['line_width'],
        title          = '',  
        style          = None,
        row_height     = standard_values['row_height'],
        show_lead_name = True,
        show_grid      = False,
        show_dc_pulse  = False,
        y_grid = 0,
        x_grid = 0,
        standard_colours = False,
        bbox = False,
        print_txt=False,
        json_dict=dict(),
        start_index=-1,
        store_configs=0,
        lead_length_in_seconds=10,
        trace_thickness_mm=None,
        trace_thickness_jitter=0.15,
        trace_dropout_rate=0.0,
        trace_dropout_length_mm=0.5,
        lead_name_gap_mm=None,
        lead_name_gap_jitter_mm=0.0,
        column_gap_mm=None,
        column_gap_jitter_mm=0.0,
        store_gridpoints=False,
        seed=-1
        ):
    #Inputs :
    #ecg - Dictionary of ecg signal with lead names as keys
    #sample_rate - Sampling rate of the ecg signal
    #lead_index - Order of lead indices to be plotted
    #columns - Number of columns to be plotted in each row
    #x_gap - gap between paper x axis border and signal plot
    #y_gap - gap between paper y axis border and signal plot
    #line_width - Width of line tracing the ecg
    #title - Title of figure
    #style - Black and white or colour
    #row_height - gap between corresponding ecg rows
    #show_lead_name - Option to show lead names or skip
    #show_dc_pulse - Option to show dc pulse
    #show_grid - Turn grid on or off
    #trace_thickness_mm - Stroke width of the ecg trace in mm of paper. None keeps the
    #                     line_width default, reproducing the upstream render exactly
    #trace_thickness_jitter - Relative amplitude of the stroke width modulation along the
    #                     trace. Only applied when trace_thickness_mm is set
    #trace_dropout_rate - Intermittent trace dropouts per cm of trace. 0 disables them and
    #                     reproduces the render exactly
    #trace_dropout_length_mm - Mean dropout length in mm of paper, drawn from an
    #                     exponential distribution
    #column_gap_mm - Blank space opened at each seam between columns of a grid layout, in
    #                     mm of paper, replacing the upstream lead-separator tick. None
    #                     keeps the columns flush and the tick drawn, reproducing the
    #                     upstream render exactly
    #column_gap_jitter_mm - Half width of an independent uniform draw applied to each
    #                     seam's gap. Only applied when column_gap_mm is set or this is > 0


    #Initialize some params
    #secs represents how many seconds of ecg are plotted
    #leads represent number of leads in the ecg
    #rows are calculated based on corresponding number of leads and number of columns

    matplotlib.use("Agg")

    #check if the ecg dict is empty
    if ecg == {}:
        return 

    secs = lead_length_in_seconds

    leads = len(lead_index)

    rows  = int(ceil(leads/columns))

    if(full_mode!='None'):
        rows+=1
        leads+=1
    
    #Grid calibration
    #Each big grid corresponds to 0.2 seconds and 0.5 mV
    #To do: Select grid size in a better way
    y_grid_size = standard_values['y_grid_size']
    x_grid_size = standard_values['x_grid_size']
    grid_line_width = standard_values['grid_line_width']
    lead_name_offset = standard_values['lead_name_offset']
    lead_fontsize = standard_values['lead_fontsize']


    #Set max and min coordinates to mark grid. Offset x_max slightly (i.e by 1 column width)

    if papersize=='':
        width = standard_values['width']
        height = standard_values['height']
    else:
        width = papersize_values[papersize][1]
        height = papersize_values[papersize][0]
    
    y_grid = standard_values['y_grid_inch'] 
    x_grid = standard_values['x_grid_inch']
    y_grid_dots = y_grid*resolution
    x_grid_dots = x_grid*resolution
 
    #row_height = height * y_grid_size/(y_grid*(rows+2))
    row_height = (height * y_grid_size/y_grid)/(rows+2)
    x_max = width * x_grid_size / x_grid
    x_min = 0
    x_gap = np.floor(((x_max - (columns*secs))/2)/0.2)*0.2
    y_min = 0
    y_max = height * y_grid_size/y_grid

    json_dict['width'] = int(width*resolution)
    json_dict['height'] = int(height*resolution)
    #Set figure and subplot sizes
    fig, ax = plt.subplots(figsize=(width, height), dpi=resolution)
   
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  
        right  = 1,  
        bottom = 0,  
        top    = 1
        )

    fig.suptitle(title)

    #Mark grid based on whether we want black and white or colour
    
    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    elif(standard_colours > 0):
        random_colour_index = standard_colours
        color_major = standard_major_colors['colour'+str(random_colour_index)]
        color_minor = standard_minor_colors['colour'+str(random_colour_index)]
        grey_random_color = random.uniform(0,0.2)
        color_line  = (grey_random_color,grey_random_color,grey_random_color)
    else:
        major_random_color_sampler_red = random.uniform(0,0.8)
        major_random_color_sampler_green = random.uniform(0,0.5)
        major_random_color_sampler_blue = random.uniform(0,0.5)

        minor_offset = random.uniform(0,0.2)
        minor_random_color_sampler_red = major_random_color_sampler_red + minor_offset
        minor_random_color_sampler_green = random.uniform(0,0.5) + minor_offset
        minor_random_color_sampler_blue = random.uniform(0,0.5) + minor_offset

        grey_random_color = random.uniform(0,0.2)
        color_major = (major_random_color_sampler_red,major_random_color_sampler_green,major_random_color_sampler_blue)
        color_minor = (minor_random_color_sampler_red,minor_random_color_sampler_green,minor_random_color_sampler_blue)
        
        color_line  = (grey_random_color,grey_random_color,grey_random_color)

    #Set grid
    #Standard ecg has grid size of 0.5 mV and 0.2 seconds. Set ticks accordingly
    
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    #Step size will be number of seconds per sample i.e 1/sampling_rate
    step = (1.0/sample_rate)

    #Resolve the trace stroke width. trace_thickness_mm is a paper length, so converting
    #it here keeps the rendered width invariant to the output resolution. Reassigning
    #line_width also carries the new thickness into the calibration pulse (x1.5) and the
    #lead separator ticks (x3), which the same stylus prints. When it is None nothing
    #below changes and the upstream render is reproduced byte for byte.
    if trace_thickness_mm is not None:
        line_width = mm_to_points(trace_thickness_mm)
    trace_jitter = trace_thickness_jitter if trace_thickness_mm is not None else 0.0
    #1 mm of paper spans x_grid_size/(x_grid_inch*25.4) data units on the x axis
    samples_per_mm = (x_grid_size/(standard_values['x_grid_inch']*25.4))*sample_rate
    trace_style = {'line_width': line_width,
                   'jitter': trace_jitter,
                   'samples_per_mm': samples_per_mm,
                   'correlation_length_mm': standard_values['trace_thickness_noise_length_mm'],
                   'rng': np.random.default_rng([abs(seed), abs(start_index)]) if trace_jitter > 0 else None,
                   'dropout_rate': trace_dropout_rate,
                   'dropout_length_mm': trace_dropout_length_mm,
                   #A stream of its own, so that turning the dropouts on never shifts the
                   #width modulation nor the global random stream that picks grid colours
                   'dropout_rng': np.random.default_rng([abs(seed), abs(start_index), 2]) if trace_dropout_rate > 0 else None
                   }

    dc_offset = 0
    if(show_dc_pulse):
        dc_offset = sample_rate*standard_values['dc_offset_length']*step
    #Iterate through each lead in lead_index array.
    y_offset = (row_height/2)
    x_offset = 0

    leads_ds = []

    leadNames_12 = configs['leadNames_12']
    tickLength = configs['tickLength']
    tickSize_step = configs['tickSize_step']

    #Vertical gap between a grid lead's name and its trace baseline. Upstream fixes it
    #at lead_name_offset + 0.2 data units (7 mm on paper). --lead_name_gap_mm overrides
    #that base distance; --lead_name_gap_jitter_mm perturbs it by a uniform half width
    #drawn ONCE per frame (its own RNG stream, keyed like the trace-width one, so it
    #never disturbs the global random draws that pick grid colours), so every grid label
    #in the frame shifts together. lead_name_gap_data stays None at the defaults and the
    #legacy expression below is left untouched, reproducing the render byte for byte.
    lead_name_gap_data = None
    if lead_name_gap_mm is not None or lead_name_gap_jitter_mm > 0:
        mm_per_data_unit = standard_values['y_grid_inch'] * 25.4 / y_grid_size
        gap_mm = lead_name_gap_mm if lead_name_gap_mm is not None else 7.0
        if lead_name_gap_jitter_mm > 0:
            gap_rng = np.random.default_rng([abs(seed), abs(start_index), 3])
            gap_mm += gap_rng.uniform(-lead_name_gap_jitter_mm, lead_name_gap_jitter_mm)
        #Floor the gap so a large downward jitter cannot ride the label into the baseline.
        lead_name_gap_data = max(gap_mm, 2.0) / mm_per_data_unit

    #Blank space between columns of a grid layout, replacing the upstream lead-separator
    #tick. Upstream places the 3x4 columns flush against each other (x_offset steps by
    #exactly `secs`) and prints a short vertical tick at every seam. --column_gap_mm opens
    #a real gap at each seam instead and suppresses the tick; --column_gap_jitter_mm makes
    #each seam's gap an independent uniform draw of half width that many mm, on its own RNG
    #stream keyed like the trace-width and lead-name-gap streams so it never disturbs the
    #global draws that pick grid colours. The same seam gaps are reused for every row, so
    #the blanks line up as clean vertical channels - the four columns come from one
    #physical printout at fixed x positions, so a per-row offset would be wrong. The gap
    #shows grid paper without a trace, which is what a real ECG with wider column spacing
    #looks like.
    #
    #The page width is held fixed: the gaps come out of the side margin and the widened
    #column block is re-centred. Each seam gap is capped so the block still clears a 0.2
    #data-unit (5 mm) margin on each side; the cap is per seam rather than a proportional
    #rescale, so one large draw does not shrink the others and decorrelate the seams.
    #
    #column_gap_cum stays None at the defaults and the legacy x_offset / x_gap / separator
    #path is left untouched, reproducing the upstream render byte for byte.
    column_gap_cum = None
    seam_gaps_mm = None
    if column_gap_mm is not None or column_gap_jitter_mm > 0:
        mm_per_data_unit_x = standard_values['x_grid_inch'] * 25.4 / x_grid_size
        n_seams = max(columns - 1, 0)
        base_mm = column_gap_mm if column_gap_mm is not None else 0.0
        if column_gap_jitter_mm > 0:
            col_gap_rng = np.random.default_rng([abs(seed), abs(start_index), 4])
            seam_gaps = np.array([max(base_mm + col_gap_rng.uniform(-column_gap_jitter_mm,
                                                                    column_gap_jitter_mm), 0.0)
                                  for _ in range(n_seams)]) / mm_per_data_unit_x
        else:
            seam_gaps = np.full(n_seams, max(base_mm, 0.0) / mm_per_data_unit_x)
        #Cap each seam so the widened, re-centred block still clears a 5 mm side margin.
        #drawn_content carries the dc_offset because every column's x_vals includes it.
        margin_reserve = 0.2
        drawn_content = columns*secs + dc_offset
        max_total_gap = max(x_max - drawn_content - 2*margin_reserve, 0.0)
        if n_seams:
            seam_gaps = np.minimum(seam_gaps, max_total_gap/n_seams)
        seam_gaps_mm = [round(g*mm_per_data_unit_x, 3) for g in seam_gaps]
        column_gap_cum = np.concatenate(([0.0], np.cumsum(seam_gaps)))
        #Re-centre the now wider block. At a total gap of 0 this is the upstream expression.
        x_gap = np.floor(((x_max - (columns*secs) - column_gap_cum[-1])/2)/0.2)*0.2

    for i in np.arange(len(lead_index)):
        current_lead_ds = dict()

        if len(lead_index) == 12:
            leadName = leadNames_12[i]
        else:
            leadName = lead_index[i]
        #y_offset is computed by shifting by a certain offset based on i, and also by row_height/2 to account for half the waveform below the axis
        if(i%columns==0):

            y_offset += row_height
        
        #x_offset will be distance by which we shift the plot in each iteration
        if(columns>1):
            x_offset = (i%columns)*secs
            
        else:
            x_offset = 0

        #Rightward shift of this column from the cumulative blank space of the seams to its
        #left. 0 for column 0 and for every layout when the gap feature is off.
        col_off = column_gap_cum[i%columns] if column_gap_cum is not None else 0.0

        #Create dc pulse wave to plot at the beginning of plot. Dc pulse will be 0.2 seconds
        x_range = np.arange(0,sample_rate*standard_values['dc_offset_length']*step + 4*step,step)
        dc_pulse = np.ones(len(x_range))
        dc_pulse = np.concatenate(((0,0),dc_pulse[2:-2],(0,0)))

        #Print lead name at .5 ( or 5 mm distance) from plot
        if(show_lead_name):
            lead_name_y = (y_offset - lead_name_offset - 0.2) if lead_name_gap_data is None \
                else (y_offset - lead_name_gap_data)
            t1 = ax.text(x_offset + x_gap + dc_offset + col_off,
                    lead_name_y,
                    leadName,
                    fontsize=lead_fontsize)
            
            if (store_text_bbox):
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent()    
                x1 = bb.x0*resolution/fig.dpi      
                y1 = bb.y0*resolution/fig.dpi   
                x2 = bb.x1*resolution/fig.dpi     
                y2 = bb.y1*resolution/fig.dpi    
                box_dict = dict()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_dict[0] = [round(x1, 2), round(json_dict['height'] - y2, 2)]
                box_dict[1] = [round(x2, 2), round(json_dict['height'] - y2, 2)]
                box_dict[2] = [round(x2, 2), round(json_dict['height'] - y1, 2)]
                box_dict[3] = [round(x1, 2), round(json_dict['height'] - y1, 2)]
                current_lead_ds["text_bounding_box"] = box_dict

        current_lead_ds["lead_name"] = leadName

        #If we are plotting the first row-1 plots, we plot the dc pulse prior to adding the waveform
        if(columns == 1 and i in np.arange(0,rows)):
            if(show_dc_pulse):
                #Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(x_range + x_offset + x_gap,
                        dc_pulse+y_offset,
                        linewidth=line_width * 1.5, 
                        color=color_line
                        )
                if (bbox):
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()                                                
                    x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                    x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
                    
                
        elif(i%columns == 0):
            if(show_dc_pulse):
                #Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(np.arange(0,sample_rate*standard_values['dc_offset_length']*step + 4*step,step) + x_offset + x_gap,
                        dc_pulse+y_offset,
                        linewidth=line_width * 1.5, 
                        color=color_line
                        )
                if (bbox):
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()                                                
                    x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                    x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi

        x_vals = np.arange(0,len(ecg[leadName])*step,step) + x_offset + dc_offset + x_gap + col_off
        y_vals = ecg[leadName] + y_offset

        trace_bb = draw_trace(ax, x_vals, y_vals, color_line, trace_style, bbox)

        if (bbox):
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = trace_bb  
            if show_dc_pulse == False or (columns == 4 and (i != 0 and i != 4 and i != 8)):                                           
                x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
            else:
                y1 = min(y1, bb.y0*resolution/fig.dpi)
                y2 = max(y2, bb.y1*resolution/fig.dpi)
                x2 = bb.x1*resolution/fig.dpi
            box_dict = dict()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            box_dict[0] = [round(x1, 2), round(json_dict['height'] - y2, 2)]
            box_dict[1] = [round(x2, 2), round(json_dict['height'] - y2, 2)]
            box_dict[2] = [round(x2, 2), round(json_dict['height'] - y1, 2)]
            box_dict[3] = [round(x1, 2), round(json_dict['height'] - y1, 2)]
            current_lead_ds["lead_bounding_box"] = box_dict
        
        st = start_index
        if columns == 4 and leadName in configs['format_4_by_3'][1]:
            st = start_index + int(sample_rate*configs['paper_len']/columns)
        elif columns == 4 and leadName in configs['format_4_by_3'][2]:
            st = start_index + int(2*sample_rate*configs['paper_len']/columns)
        elif columns == 4 and leadName in configs['format_4_by_3'][3]:
            st = start_index + int(3*sample_rate*configs['paper_len']/columns)
        current_lead_ds["start_sample"] = st
        current_lead_ds["end_sample"]= st + len(ecg[leadName])
        current_lead_ds["plotted_pixels"] = []
        for j in range(len(x_vals)):
            xi, yi = x_vals[j], y_vals[j]
            xi, yi = ax.transData.transform((xi, yi))
            yi = json_dict['height'] - yi
            current_lead_ds['plotted_pixels'].append([round(xi, 2), round(yi, 2)])

        leads_ds.append(current_lead_ds)

        #The separator tick is replaced by the blank space when the gap feature is on.
        if columns > 1 and (i+1)%columns != 0 and column_gap_cum is None:
            sep_x = [len(ecg[leadName])*step + x_offset + dc_offset + x_gap] * round(tickLength*y_grid_dots)
            sep_x = np.array(sep_x)
            sep_y = np.linspace(y_offset - tickLength/2*y_grid_dots*tickSize_step, y_offset + tickSize_step*y_grid_dots*tickLength/2, len(sep_x))
            ax.plot(sep_x, sep_y, linewidth=line_width * 3, color=color_line)

    #Plotting longest lead for 12 seconds
    if(full_mode!='None'):
        current_lead_ds = dict()
        if(show_lead_name):
            t1 = ax.text(x_gap + dc_offset, 
                    row_height/2-lead_name_offset, 
                    full_mode, 
                    fontsize=lead_fontsize)
            
            if (store_text_bbox):
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent(renderer = fig.canvas.renderer)
                x1 = bb.x0*resolution/fig.dpi      
                y1 = bb.y0*resolution/fig.dpi   
                x2 = bb.x1*resolution/fig.dpi     
                y2 = bb.y1*resolution/fig.dpi           
                box_dict = dict()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_dict[0] = [round(x1, 2), round(json_dict['height'] - y2, 2)]
                box_dict[1] = [round(x2, 2), round(json_dict['height'] - y2, 2)]
                box_dict[2] = [round(x2, 2), round(json_dict['height'] - y1, 2)]
                box_dict[3] = [round(x1, 2), round(json_dict['height'] - y1, 2)]
                current_lead_ds["text_bounding_box"] = box_dict                
            current_lead_ds["lead_name"] = full_mode

        if(show_dc_pulse):
            t1 = ax.plot(x_range + x_gap,
                    dc_pulse + row_height/2-lead_name_offset + 0.8,
                    linewidth=line_width * 1.5, 
                    color=color_line
                    )
            
            if (bbox):
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()                                                
                    x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                    x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
        
        dc_full_lead_offset = 0 
        if(show_dc_pulse):
            dc_full_lead_offset = sample_rate*standard_values['dc_offset_length']*step
        
        x_vals = np.arange(0,len(ecg['full'+full_mode])*step,step) + x_gap + dc_full_lead_offset
        y_vals = ecg['full'+full_mode] + row_height/2-lead_name_offset + 0.8

        trace_bb = draw_trace(ax, x_vals, y_vals, color_line, trace_style, bbox)

        if (bbox):
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = trace_bb  
            if show_dc_pulse == False:                                           
                x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
            else:
                y1 = min(y1, bb.y0*resolution/fig.dpi)
                y2 = max(y2, bb.y1*resolution/fig.dpi)
                x2 = bb.x1*resolution/fig.dpi

            box_dict = dict()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            box_dict[0] = [round(x1, 2), round(json_dict['height'] - y2, 2)]
            box_dict[1] = [round(x2, 2), round(json_dict['height'] - y2, 2)]
            box_dict[2] = [round(x2, 2), round(json_dict['height'] - y1, 2)]
            box_dict[3] = [round(x1, 2), round(json_dict['height'] - y1, 2)]
            current_lead_ds["lead_bounding_box"] = box_dict
        current_lead_ds["start_sample"] = start_index
        current_lead_ds["end_sample"] = start_index + len(ecg['full'+full_mode])
        current_lead_ds['plotted_pixels'] = []
        for i in range(len(x_vals)):
            xi, yi = x_vals[i], y_vals[i]
            xi, yi = ax.transData.transform((xi, yi))
            yi = json_dict['height'] - yi
            current_lead_ds['plotted_pixels'].append([round(xi, 2), round(yi, 2)])
        leads_ds.append(current_lead_ds)



    head, tail = os.path.split(rec_file_name)
    rec_file_name = os.path.join(output_dir, tail)

    #printed template file
    if print_txt:
        x_offset = 0.05
        y_offset = int(y_max)
        printed_text, attributes, flag = generate_template(full_header_file)

        if flag:
            for l in range(0, len(printed_text), 1):
        
                for j in printed_text[l]:
                    curr_l = ''
                    if j in attributes.keys():
                        curr_l += str(attributes[j])
                    ax.text(x_offset, y_offset, curr_l, fontsize=lead_fontsize)
                    x_offset += 3

                y_offset -= 0.5
                x_offset = 0.05
        else:
            for line in printed_text:
                ax.text(x_offset, y_offset, line, fontsize=lead_fontsize)
                y_offset -= 0.5

    #change x and y res
    ax.text(2, 0.5, '25mm/s', fontsize=lead_fontsize)
    ax.text(4, 0.5, '10mm/mV', fontsize=lead_fontsize)
    
    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,x_grid_size))    
        ax.set_yticks(np.arange(y_min,y_max,y_grid_size))
        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        #set grid line style
        ax.grid(which='major', linestyle='-', linewidth=grid_line_width, color=color_major)
        
        ax.grid(which='minor', linestyle='-', linewidth=grid_line_width, color=color_minor)
        
        if store_configs == 2:
            json_dict['grid_line_color_major'] = [round(x*255., 2) for x in color_major]
            json_dict['grid_line_color_minor'] = [round(x*255., 2) for x in color_minor]
            json_dict['ecg_plot_color'] = [round(x*255., 2) for x in color_line]
    else:
        ax.grid(False)

    if store_configs == 2:
        json_dict['trace_thickness_mm'] = round(line_width*25.4/72.0, 4)
        json_dict['trace_thickness_jitter'] = trace_jitter
        json_dict['trace_dropout_rate'] = trace_dropout_rate
        json_dict['trace_dropout_length_mm'] = trace_dropout_length_mm
        if seam_gaps_mm is not None:
            #The realized per-seam blank width in mm; the requested flag is not
            #recoverable from the image once the per-frame jitter is drawn.
            json_dict['column_gap_mm'] = seam_gaps_mm

    if store_gridpoints:
        #Top-level keys (not nested under 'leads'), computed in the unpadded render
        #frame - same convention as plotted_pixels above, so PaperCrumple.crumple and
        #CameraSensor.sensor's annotation transforms pick them up with a plain key
        #check. Independent of show_grid: the lattice exists whether or not gridlines
        #are actually drawn. gridpoints_mask is deliberately NOT set here - it depends
        #on the FINAL frame after crumple/augment, and is computed once, at the end of
        #the whole pipeline, in gen_ecg_image_from_data.py.
        gridpoints, gridpoints_shape, gridpoints_reference_hw = build_gridpoints(
            ax, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size,
            json_dict['height'])
        json_dict['gridpoints'] = gridpoints
        json_dict['gridpoints_shape'] = gridpoints_shape
        json_dict['gridpoints_mm_per_step'] = GRIDPOINT_MM_PER_STEP
        json_dict['gridpoints_reference_hw'] = gridpoints_reference_hw

    plt.savefig(os.path.join(output_dir,tail +'.png'),dpi=resolution)
    plt.close(fig)
    plt.clf()
    plt.cla()

    if pad_inches!=0:
        
        ecg_image = Image.open(os.path.join(output_dir,tail +'.png'))
        
        right = pad_inches * resolution
        left = pad_inches * resolution
        top = pad_inches * resolution
        bottom = pad_inches * resolution
        width, height = ecg_image.size
        new_width = width + right + left
        new_height = height + top + bottom
        result_image = Image.new(ecg_image.mode, (new_width, new_height), (255, 255, 255))
        result_image.paste(ecg_image, (left, top))
        
        result_image.save(os.path.join(output_dir,tail +'.png'))

        plt.close('all')
        plt.close(fig)
        plt.clf()
        plt.cla()

    json_dict["leads"] = leads_ds

    return x_grid_dots,y_grid_dots
       