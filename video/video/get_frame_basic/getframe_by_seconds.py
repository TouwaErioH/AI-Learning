import ffmpeg
import numpy
import cv2
import sys


def read_frame_as_jpeg(in_file, frame_num):
    """
    指定帧数读取任意帧
    """
    out, err = (
        ffmpeg.input(in_file)
              .filter('select', 'gte(n,{})'.format(frame_num))
              .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
              .run(capture_stdout=True)
    )
    return out


def get_video_info(in_file):
    """
    获取视频基本信息
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)


def improve(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 最终转为二值图
    # ret, image = cv2.threshold(image, 127, 200, cv2.THRESH_BINARY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image


if __name__ == '__main__':
    file_path = 'video.mp4'
    video_info = get_video_info(file_path)
    total_duration = video_info['duration']
    avg_frame_rate = video_info['avg_frame_rate']

    i = 0
    gap = 0
    gap_frame = 38  # 每季度持续38帧
    while gap < float(total_duration)*30:
        if i == 0:
            gap += 2*gap_frame + 1
        else:
            gap += 4*gap_frame
        out = read_frame_as_jpeg(file_path, (gap/30)*(30000/1001))
        image_array = numpy.asarray(bytearray(out), dtype="uint8")
        if image_array.any() and (1965 + i <= 2019):
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            #image = improve(image)
            cv2.imwrite(str(1965 + i) + 'final Q4.jpg', image)
        i += 1
        
