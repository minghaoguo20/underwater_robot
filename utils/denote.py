def denote_msg(denote_path, flag_msg_number, now, cam_angle=None, cam_err=None, lines=None, cal=None, line_nearest=None, cal_nearesr=None):
    f = open(denote_path, 'a+')
    f.write(f"{now.strftime('%Y%m%d-%H:%M:%S')} flag_msg_number = {flag_msg_number}\n")
    if cam_angle is not None and cam_err is not None:
        f.write(f"cam_angle: {cam_angle}, cam_err: {cam_err}\n")

    if line_nearest is not None:
        f.write("line_nearest:\n")
        f.write("no line_nearest\n") if line_nearest.shape[0] == 0 else f.write(str(line_nearest))
    if cal_nearesr is not None:
        f.write("cal_nearesr:\n")
        f.write("no cal_nearesr\n") if cal_nearesr.shape[0] == 0 else f.write(str(cal_nearesr))

    if lines is not None:
        f.write("lines:\n")
        f.write("no lines\n") if lines.shape[0] == 0 else f.write(str(lines))
    if cal is not None:
        f.write("cal:\n")
        f.write("no cal\n") if cal.shape[0] == 0 else f.write(str(cal))
    f.write("\n\n")
    f.close()