from LaneFinder.finder import LaneFinder
import cv2
import moviepy.editor as mpy
from Camera.UdacityAdvLane import UdacityAdvLane as Ucam


input_file = "project_video.mp4"
output_file = "augmented_project_video.mp4"

udacam = Ucam(input_file)

# We need to create iterator from udacam which will be used in below
# function to get frames from camera
cam_iterator = udacam.__iter__()


lf = LaneFinder(udacam)


def frame_generator(t):
    """This function generates annotated frames for video"""

    # Get next frame
    frame = next(cam_iterator)

    # Run frame through lane finder
    lf.apply(frame)
    # Annotate lane
    lane = lf.draw_lane(color=(0, 255, 0), y_range=(100, 500))
    unwarped_lane = lf.camera.apply_pipeline_inverse(lane)
    unwarped_annotated_lane = cv2.addWeighted(lf.camera.latest_undistorted, 1,
                                              unwarped_lane, 0.5, 0)

    # Insert small warped image onto big image
    warped_search_area = lf.visualize_finder()
    unwarped_annotated_lane[:522, :266] = 0  # Cut black hole on left top corner
    unwarped_annotated_lane[:512, :256] = warped_search_area

    # Add lane curvature and offset readings
    curve_rad = lf.curve_radius
    offset = lf.lane_offset
    cv2.putText(unwarped_annotated_lane,
                "Curve Radius: {:.0f}m".format(curve_rad), (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(unwarped_annotated_lane, "Off Center:   {:.2f}m".format(offset),
                (300, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


    # Need to convert from BGR to RGB to get colors right in video
    unwarped_annotated_lane = cv2.cvtColor(unwarped_annotated_lane, cv2.COLOR_BGR2RGB)
    return unwarped_annotated_lane

# Write video file to disk
augmented_video = mpy.VideoClip(frame_generator, duration=udacam.clip.duration)
augmented_video.write_videofile(output_file, fps=udacam.clip.fps)