import numpy as np


def check_pog(t,
              vm,
              vleg,
              bearing_angle,
              lead_angle,
              alpha_total,
              body_rates):
    vleg_tolerance = 0.01
    total_aoa_tolerance = 0.01
    rate_tolerance = 0.01

    # Check if Pitchover Sequence is Complete
    ######################################################
    # Compute current vleg
    currentVLEG = np.arctan2(vm[2], np.linalg.norm(vm[:2]))

    # Check if current vleg is within desired tolerance
    vlegDiff = vleg - currentVLEG
    vlegCheck = abs(vlegDiff) <= vleg_tolerance

    # Check if total aoa is within desired tolerance
    aoaCheck = alpha_total <= total_aoa_tolerance

    # Check if body rates are within desired tolerance
    rateCheck = (np.abs(body_rates) <= rate_tolerance).all()

    # print(vlegCheck, aoaCheck, rateCheck)

    if vlegCheck and aoaCheck and rateCheck:
        complete = True
    else:
        complete = False
    ######################################################
    return complete


def pog(t,
        desired_vleg,
        desired_bearing_angle,
        alphaTotal,
        altm,
        # C_body2enu,
        # gravity,
        lead_angle,
        vm,
        body_rates,
        complete=False) -> tuple:

    altitudePSS = 0  # pitch sequence start altitude

    if altm >= altitudePSS and not complete:
        vleg = desired_vleg
        bearing_angle = desired_bearing_angle

        complete = check_pog(t, vm, vleg, bearing_angle, lead_angle, alphaTotal, body_rates)

        # Compute desired flight path vector
        fp = np.array([np.cos(vleg) * np.sin(bearing_angle),
                       np.cos(vleg) * np.cos(bearing_angle),
                       np.sin(vleg)])

        # Compute missile velocity vector
        if np.linalg.norm(vm) > 0:
            uvm = vm / np.linalg.norm(vm)
        else:
            uvm = np.zeros(3)

        # Compute angle between missile velocity and EN-plane
        vma_vert = np.arctan2(vm[2], np.sqrt(vm[0]**2 + vm[1]**2))

        # Compute heading error
        heading_error = abs(vleg - vma_vert)

        # Compute angle between missile velocity and NU-plane
        vma_horz = np.arctan2(vm[0], np.sqrt(vm[1]**2 + vm[2]**2))

        # Define desired flight path angle on EN-plane
        fpa_horz = bearing_angle + lead_angle

        # Compute bearing error
        bearing_error = abs(fpa_horz - vma_horz)

        # print(complete, np.degrees(vleg), np.degrees(vma_vert))
        # print(complete, np.degrees(vma_horz), np.degrees(fpa_horz))
        pass

        gainK = 0.0986
        gainKh = 0

        guide_law = "mr_pitchover"
        match guide_law:
            case "mr_pitchover":
                # Compute "vertical" fp angle rate (proportional
                # to flight path angle error)
                fpaRate = gainK * heading_error

                # Compute guidance command
                ac3 = np.linalg.norm(vm) * fpaRate

                # Compute "horizontal" fp angle rate
                leadRate = gainKh * bearing_error

                # Compute guidance command
                ac2 = np.linalg.norm(vm) * leadRate

                # Rotate command from body frame to ENU
                a_c = np.array([0, ac2, ac3])
            # case "simple":
            #     gainK = gainK * gravity
            #     a_c = gainK * (fp - np.dot(fp, uvm) * uvm)
            case _:
                # no guidance (ballistic)
                a_c = np.zeros(3)

        return (complete, a_c)
    return (complete, np.zeros(3))
