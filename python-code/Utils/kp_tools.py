# kp_tools.py ---
#
# Filename: kp_tools.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Mon Aug 31 10:59:33 2015 (+0200)
# Version:
# Package-Requires: ()
# Last-Updated: Mon Jul  4 05:57:02 2016 (-0700)
#           By: Kwang Moo Yi
#     Update #: 108
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

from __future__ import print_function

import numpy

# Keypoint List Structure Index Info

# opencv
IDX_X, IDX_Y, IDX_SIZE, IDX_ANGLE, IDX_RESPONSE, IDX_OCTAVE = (
    0, 1, 2, 3, 4, 5)  # , IDX_CLASSID not used
# vgg affine
IDX_a, IDX_b, IDX_c = (6, 7, 8)
# vlfeat Affine [A0, A1; A2, A3]?
# NOTE the row-major colon-major adaptation here
IDX_A0, IDX_A2, IDX_A1, IDX_A3 = (9, 10, 11, 12)


def update_affine(kp):
    # Compute A0, A1, A2, A3
    S = numpy.asarray([[kp[IDX_a], kp[IDX_b]],
                       [kp[IDX_b], kp[IDX_c]]])
    invS = numpy.linalg.inv(S)
    a = numpy.sqrt(invS[0, 0])
    b = invS[0, 1] / max(a, 1e-18)
    A = numpy.asarray([[a, 0],
                       [b, numpy.sqrt(max(invS[1, 1] - b**2, 0))]])

    # We need to rotate first!
    cos_val = numpy.cos(numpy.deg2rad(kp[IDX_ANGLE]))
    sin_val = numpy.sin(numpy.deg2rad(kp[IDX_ANGLE]))
    R = numpy.asarray([[cos_val, -sin_val],
                       [sin_val, cos_val]])

    A = numpy.dot(A, R)

    kp[IDX_A0] = A[0, 0]
    kp[IDX_A1] = A[0, 1]
    kp[IDX_A2] = A[1, 0]
    kp[IDX_A3] = A[1, 1]

    return kp


def loadKpListFromTxt(kp_file_name):

    # Open keypoint file for read
    kp_file = open(kp_file_name, 'rb')

    # skip the first two lines
    kp_line = kp_file.readline()
    kp_line = kp_file.readline()

    kp_list = []
    num_elem = -1
    while True:
        # read a line from file
        kp_line = kp_file.readline()
        # check EOF
        if not kp_line:
            break
        # split read information
        kp_info = kp_line.split()
        parsed_kp_info = []
        for idx in xrange(len(kp_info)):
            parsed_kp_info += [float(kp_info[idx])]
        parsed_kp_info = numpy.asarray(parsed_kp_info)

        if num_elem == -1:
            num_elem = len(parsed_kp_info)
        else:
            assert num_elem == len(parsed_kp_info)

        # IMPORTANT: make sure this part corresponds to the one in
        # opencv_kp_list_2_kp_list

        # check if we have all the kp list info
        if len(parsed_kp_info) == 6:       # if we only have opencv info
            # Compute a,b,c for vgg affine
            a = 1. / (parsed_kp_info[IDX_SIZE]**2)
            b = 0.
            c = 1. / (parsed_kp_info[IDX_SIZE]**2)
            parsed_kp_info = numpy.concatenate((parsed_kp_info, [a, b, c]))

        if len(parsed_kp_info) == 9:       # if we don't have the Affine warp
            parsed_kp_info = numpy.concatenate(
                (parsed_kp_info, numpy.zeros((4,))))
            parsed_kp_info = update_affine(parsed_kp_info)

        assert len(parsed_kp_info) == 13  # make sure we have everything!

        kp_list += [parsed_kp_info]

    # Close keypoint file
    kp_file.close()

    return kp_list


def saveKpListToTxt(kp_list, orig_kp_file_name, kp_file_name):

    # # Open keypoint file to read the dim value
    # if orig_kp_file_name is not None:
    #     kp_file = open(orig_kp_file_name,'rb')
    #     kp_line = kp_file.readline()
    #     kp_file.close()
    # else:
    #     # print('First line is hard coded to 13! is this wise?')
    kp_line = '13\n'            # first line 13 to indicate we have the full

    # Open keypoint file for write
    kp_file = open(kp_file_name, 'wb')

    # write the first line
    kp_file.write(kp_line)

    # write the number of kp in second line
    kp_file.write('{}\n'.format(len(kp_list)))

    for kp in kp_list:

        # Make sure we have all info for kp
        assert len(kp) == 13

        # Form the string to write
        write_string = ""
        for kp_elem, _i in zip(kp, range(len(kp))):
            if _i == 5:         # in case of the octave
                write_string += str(numpy.int32(kp_elem)) + " "
            else:
                write_string += str(kp_elem) + " "
        write_string += "\n"

        # Write the string
        kp_file.write(write_string)

    # Close keypoint file
    kp_file.close()


#
# kp_tools.py ends here
