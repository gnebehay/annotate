#!/usr/bin/env python

import cv2
import numpy as np
import os
import sequences
import sys

NUM_COLUMNS = 6
FRAME_NO = 0
CENTER_X = 1
CENTER_Y = 2
SIZE_X = 3
SIZE_Y = 4
ANGLE = 5

def array_to_rotated_rect(arr):
    return ((arr[CENTER_X],arr[CENTER_Y]),(arr[SIZE_X],arr[SIZE_Y]),arr[ANGLE])

def rrect_to_arr(frame, rrect):
    return np.array((frame, rrect[0][0], rrect[0][1], rrect[1][0], rrect[1][1], rrect[2]))

def get_interpolated_rrect(frame, annot):
    #Extract current annotation
    entry = annot[annot[:,FRAME_NO] == frame,:].squeeze()

    rrect = None
    if not entry.size == 0:
        rrect = array_to_rotated_rect(entry)
        print('Frame ' + str(frame) + ' is a key frame.')
    else:

        #Perform linear interpolation
        interpol_possible = True

        entries_before = annot[annot[:,FRAME_NO] < frame,:]
        if entries_before.size > 0:
            previous_entry = entries_before[-1,:]
        else:
            interpol_possible = False

        entries_after = annot[annot[:,FRAME_NO] > frame,:]
        if entries_after.size > 0:
            next_entry = entries_after[0,:]
        else:
            interpol_possible = False

        if interpol_possible:
            print('Frame ' + str(frame) + ' is an interpolated frame.')
            a = previous_entry[FRAME_NO]
            b = next_entry[FRAME_NO]

            if abs(previous_entry[ANGLE] - next_entry[ANGLE]) >= 180:
                if previous_entry[ANGLE] < next_entry[ANGLE]:
                    previous_entry[ANGLE] += 360
                else:
                    next_entry[ANGLE] += 360

            beta = (frame - a) / (b - a)
            alpha = 1 - (frame - a) / (b - a)

            #This even gets the frame number right!
            interpol = alpha * previous_entry  + beta * next_entry

            rrect = array_to_rotated_rect(interpol)

    if rrect is None:
        print('No annotation for frame ' + str(frame))

    return rrect

def update_annot(annot, frame, rrect):
    #Create new annot with
    entry = annot[annot[:,FRAME_NO] == frame,:].squeeze()
    new_entry = rrect_to_arr(frame, rrect)
    if not entry.size == 0:
        annot[annot[:,FRAME_NO] == frame,:] = new_entry
    #Insert annot
    else:
        annot = np.vstack([annot, new_entry])

        #re-sort
        annot = annot[annot[:,0].argsort()]

    return annot

def annotate(sequence):

    annotation_file = os.path.join(sequence.directory, 'annotation')

    if os.path.exists(annotation_file):
        print('Using annotation file.')
        annot = np.atleast_2d(np.genfromtxt(annotation_file, delimiter=','))
        if annot.shape[1] != NUM_COLUMNS:
            raise Exception("Number of column in annotation file is wrong.")
    elif sequence.gt is not None:
        print('Using groundtruth.')
        gt = np.copy(sequence.gt)
        gt[:,:2] = gt[:,:2] + gt[:,2:] / 2

        frames = np.atleast_2d(np.array(range(sequence.num_frames))).transpose()
        angles = np.zeros((sequence.num_frames,1))

        annot = np.hstack((frames,gt,angles))
    else:
        annot = np.empty((0,6))
    
    frame = 0

    while True:

        #Read image
        im_path = sequence.im_list[frame]
        if not os.path.exists(im_path):
            raise Exception(im_path + ' does not exist')
        im = cv2.imread(im_path)
        im_draw = np.copy(im)

        rrect = get_interpolated_rrect(frame, annot)

        if rrect is not None:
            box = cv2.cv.BoxPoints(rrect)
            box = np.int0(box)
            cv2.drawContours(im_draw,[box], contourIdx=0,color=(0,0,255),thickness=2)

            #Draw angle
            center = (int(rrect[0][0]), int(rrect[0][1]))
            half_height = rrect[1][1] / 2
            angle = rrect[2] - 90
            x2 = int(center[0] + half_height * np.cos(np.radians(angle)))
            y2 = int(center[1] + half_height * np.sin(np.radians(angle)))
            cv2.line(im_draw, center, (x2,y2), (0,0,255), thickness=2)

        #Show image, wait for key
        cv2.imshow(sequence.name, im_draw)
        key = cv2.waitKey(0)
        key = chr(key & 255)

        #create
        if key == 'i':
            print 'Inserting keyframe at frame' + str(frame) + '.'
            annot = annot[annot[:,FRAME_NO] != frame,:]

            new_entry = np.array([frame, sequence.shape[1] / 2, sequence.shape[0] / 2, sequence.shape[1] / 2, sequence.shape[0] / 2, 0])

            annot = np.vstack([annot, new_entry])

            #re-sort
            annot = annot[annot[:,0].argsort()]

        #Edit
        if key == 'e':
            print 'Entering edit mode for frame ' + str(frame) + '.'

            global annot
            global frame
            global mouse_is_down
            global old_x
            global old_y
            global rrect
            global im
            mouse_is_down = False

            def onMouse(event, x, y, flags, param):
                global annot
                global frame
                global mouse_is_down
                global old_x
                global old_y
                global rrect
                global im

                mouse_down = bool(event & cv2.EVENT_LBUTTONDOWN)
                mouse_up = bool(event & cv2.EVENT_LBUTTONUP)
                alt_pressed = bool(flags & cv2.EVENT_FLAG_ALTKEY)
                ctrl_pressed = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

                if mouse_down:
                    mouse_is_down = True
                    old_x = x
                    old_y = y

                if mouse_is_down:
                    #Display what happens

                    delta_x = x - old_x
                    delta_y = y - old_y

                    #Rotation
                    if alt_pressed:
                        new_angle = rrect[2] - delta_y
                        rrect_tmp = (rrect[0], rrect[1], new_angle)
                        box = cv2.cv.BoxPoints(rrect_tmp)
                        box = np.int0(box)
                        im_draw = np.copy(im)
                        cv2.drawContours(im_draw,[box], contourIdx=0,color=(0,0,255),thickness=2)
                        cv2.imshow(sequence.name, im_draw)

                    elif ctrl_pressed:
                        new_size = (rrect[1][0] + delta_x,rrect[1][1] - delta_y)
                        rrect_tmp = (rrect[0], new_size, rrect[2])
                        box = cv2.cv.BoxPoints(rrect_tmp)
                        box = np.int0(box)
                        im_draw = np.copy(im)
                        cv2.drawContours(im_draw,[box], contourIdx=0,color=(0,0,255),thickness=2)
                        cv2.imshow(sequence.name, im_draw)

                    else:
                        new_loc = (rrect[0][0] + delta_x,rrect[0][1] + delta_y)

                        rrect_tmp = (new_loc, rrect[1], rrect[2])
                        box = cv2.cv.BoxPoints(rrect_tmp)
                        box = np.int0(box)
                        im_draw = np.copy(im)
                        cv2.drawContours(im_draw,[box], contourIdx=0,color=(0,0,255),thickness=2)
                        cv2.imshow(sequence.name, im_draw)

                if mouse_up:
                    mouse_is_down = False
                    rrect = rrect_tmp
                    annot = update_annot(annot, frame, rrect)
                    print 'Recognized an action with start ', old_x, old_y, 'and end',x,y
                    box = cv2.cv.BoxPoints(rrect)
                    box = np.int0(box)
                    im_draw = np.copy(im)
                    cv2.drawContours(im_draw,[box], contourIdx=0,color=(0,0,255),thickness=2)
                    cv2.imshow(sequence.name, im_draw)
                    #End of action
                    #TODO
                    #Clear x,y


                print 'alt_pressed', alt_pressed, 'ctrl_pressed',ctrl_pressed, 'mouse_down',mouse_down

            cv2.setMouseCallback(sequence.name, onMouse)


        #Delete
        if key == 'x':
            print 'Deleting keyframe ' + str(frame) + '.'
            annot = annot[annot[:,FRAME_NO] != frame,:]

        #Standard movement
        if key == 'S' or key == ' ':
            frame += 1
        if key == 'Q':
            frame -= 1 #Go back one frame
        if key == 'P':
            frame = 0
        if key == 'W':
            frame = sequence.num_frames-1
        if key == 'm':
            frame = sequence.num_frames / 2
        if key == 'V':
            frame += sequence.num_frames / 100
        if key == 'U':
            frame -= sequence.num_frames / 100
        if key == 'q':
            break


        #Don't go out of bounds
        frame = max(frame, 0)
        frame = min(frame, sequence.num_frames-1)

    #What a wonderful line.
    polygon = np.array([np.array(cv2.cv.BoxPoints(get_interpolated_rrect(i,annot))).flatten() for i in xrange(sequence.num_frames)])

    #Save this as the new groundtruth
    polygon_file = os.path.join(sequence.directory, 'groundtruth.txt')

    np.savetxt(polygon_file, polygon, fmt='%.2f', delimiter=',')

    #annotation_file = os.path.join(annot_folder, prop + '.label')
    np.savetxt(annotation_file, annot, fmt='%.2f', delimiter=',')

if __name__ == "__main__":
    print 'start'
    top_dir = sys.argv[1]
    seq_name = sys.argv[2]
    seqs = sequences.list_sequences(top_dir)
    seq = seqs[seq_name]
    seq.load()
    annotate(seq)
