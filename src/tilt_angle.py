import numpy as np

def compute_best_tilt(
    loss_map,
    degrees=False,
    angle_mode='signed90',
    return_m=False,
    return_b=False,
    return_coords=False,
    mask_zeros=True,
    fit_mode=None  # New argument: None (auto), 'row', or 'col'
):
    """
    Fit the low-loss ridge either row-wise or column-wise, pick the better R²,
    and return its tilt angle plus optional slope/intercept and the
    corresponding minima coords, under two angle conventions.

    Parameters
    ----------
    loss_map : (n,n) or (n,n,1) array
        2D loss grid. Zeros are treated as 'ignore' if mask_zeros=True.
    degrees : bool
        If True, angle is returned in degrees (otherwise radians).
    angle_mode : str, {'signed90', 'unsigned180'}
        'signed90'   : angles in [-90°, +90°], positive clockwise from +Y.
        'unsigned180': angles in [0°, 180°],    positive clockwise from +Y.
    return_m : bool
        If True, return the slope m of the chosen fit.
    return_b : bool
        If True, return the intercept b of the chosen fit.
    return_coords : bool
        If True, return the minima coordinates as a tuple:
          - (ys, min_cols) if row-wise was chosen
          - (xs, min_rows) if column-wise was chosen
    mask_zeros : bool
        If True, zeros in the map are replaced with +∞ so they
        don’t count as minima. Set to False if zeros *are*
        your true ridge (e.g. synthetic tests).
    fit_mode : None, 'row', or 'col'
        If None, automatically choose the better fit (default).
        If 'row', force row-wise fit. If 'col', force column-wise fit.

    Returns
    -------
    alpha : float
        Tilt angle under the chosen convention (see `angle_mode`).
    m_chosen : float, optional
        Slope of the chosen fit.
    b_chosen : float, optional
        Intercept of the chosen fit.
    coords : tuple, optional
        The minima coords:
          * row-wise: (ys, min_cols)
          * col-wise: (xs, min_rows)
    which : str
        'row' or 'col', indicating which fit was used.
    """
    # 1) squeeze to (n,n)
    arr = np.asarray(loss_map)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:,:,0]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input must be shape (n,n) or (n,n,1)")
    n = arr.shape[0]

    # 2) mask zeros if desired
    if mask_zeros:
        masked = np.where(arr!=0, arr, np.inf)
    else:
        masked = arr.copy()

    # 3) row-wise minima: xs_row[y] = column of min in row y
    valid_rows = ~np.all(masked==np.inf, axis=1)
    ys_row = np.nonzero(valid_rows)[0]
    xs_row = np.argmin(masked[valid_rows,:], axis=1)

    # 4) col-wise minima: ys_col[x] = row of min in col x
    valid_cols = ~np.all(masked==np.inf, axis=0)
    xs_col = np.nonzero(valid_cols)[0]
    ys_col = np.argmin(masked[:,valid_cols], axis=0)

    # 5) fit row-wise: x = m_row*y + b_row
    if len(ys_row) >= 2:
        m_row, b_row = np.polyfit(ys_row, xs_row, 1)
        pred_r = m_row*ys_row + b_row
        ss_res_r = np.sum((xs_row - pred_r)**2)
        ss_tot_r = np.sum((xs_row - xs_row.mean())**2)
        r2_row = 1.0 if ss_tot_r==0 else 1 - ss_res_r/ss_tot_r
    else:
        m_row, b_row, r2_row = np.nan, np.nan, -np.inf

    # 6) fit col-wise: y = m_col*x + b_col
    if len(xs_col) >= 2:
        m_col, b_col = np.polyfit(xs_col, ys_col, 1)
        pred_c = m_col*xs_col + b_col
        ss_res_c = np.sum((ys_col - pred_c)**2)
        ss_tot_c = np.sum((ys_col - ys_col.mean())**2)
        r2_col = 1.0 if ss_tot_c==0 else 1 - ss_res_c/ss_tot_c
    else:
        m_col, b_col, r2_col = np.nan, np.nan, -np.inf

    # 7) choose fit mode
    if fit_mode is None:
        # auto: pick the better R^2
        if r2_row >= r2_col:
            which     = 'row'
            m_chosen  = m_row
            b_chosen  = b_row
            # bearing from +Y axis, clockwise: atan2(dx, dy) = atan(m_row)
            alpha_raw = np.arctan(m_row)
            coords    = (ys_row, xs_row)
        else:
            which     = 'col'
            m_chosen  = m_col
            b_chosen  = b_col
            # slope dy/dx = m_col, so dx/dy = 1/m_col, bearing = atan2(dx,dy)
            alpha_raw = np.arctan2(1.0, m_col)
            coords    = (xs_col, ys_col)
    elif fit_mode == 'row':
        which     = 'row'
        m_chosen  = m_row
        b_chosen  = b_row
        alpha_raw = np.arctan(m_row)
        coords    = (ys_row, xs_row)
    elif fit_mode == 'col':
        which     = 'col'
        m_chosen  = m_col
        b_chosen  = b_col
        alpha_raw = np.arctan2(1.0, m_col)
        coords    = (xs_col, ys_col)
    else:
        raise ValueError("fit_mode must be None, 'row', or 'col'")

    # 8) apply angle convention
    if angle_mode == 'signed90':
        # map into [-π/2, +π/2]
        if alpha_raw > np.pi/2:
            alpha_raw -= np.pi
    elif angle_mode == 'unsigned180':
        # map into [0, π]
        if alpha_raw < 0:
            alpha_raw += np.pi
    else:
        raise ValueError("angle_mode must be 'signed90' or 'unsigned180'")

    alpha = np.degrees(alpha_raw) if degrees else alpha_raw

    # assemble outputs
    out = [alpha]
    if return_m:      out.append(m_chosen)
    if return_b:      out.append(b_chosen)
    if return_coords: out.append(coords)
    out.append(which)
    return tuple(out)
