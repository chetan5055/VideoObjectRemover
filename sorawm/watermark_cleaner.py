from pathlib import Path

from cv2.gapi import imgproc

from sorawm.utils.video_utils import VideoLoader

from sorawm.utils.watermark_utls import (
    detect_watermark,
    get_bounding_box,
    h_tmpl,
    w_tmpl,
)

# based on the sora tempalte to detect the whole, and then got the icon part area.


class SoraWaterMarkCleaner:
    def __init__(self, input_video_path: Path, output_video_path: Path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.input_video_loader = VideoLoader(self.input_video_path)


if __name__ == "__main__":
    from tqdm import tqdm
    import cv2
    from sorawm.configs import OUTPUT_DIR

    video_path = Path("resources/dog_vs_sam.mp4")

    cleaner = SoraWaterMarkCleaner(video_path, video_path)

    for idx, frame in enumerate(
        tqdm(cleaner.input_video_loader, desc="Processing frames")
    ):
        mask, detections = detect_watermark(frame)

        if detections:
            img_with_boxes = frame.copy()

            # 遍历所有检测结果，为每个检测画框
            for det_idx, detection in enumerate(detections):
                # detection 现在是 (x, y, w, h)
                x, y, w, h = detection

                # 画单个检测框（绿色）
                cv2.rectangle(
                    img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2  # 绿色
                )

                # 标注检测编号
                cv2.putText(
                    img_with_boxes,
                    f"#{det_idx + 1}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # 画总边界框（红色）
            bbox = get_bounding_box(detections, w_tmpl, h_tmpl)
            min_x, min_y, max_x, max_y = bbox
            cv2.rectangle(
                img_with_boxes,
                (min_x, min_y),
                (max_x, max_y),
                (0, 0, 255),  # 红色
                3,  # 更粗的线
            )

            # 标注总检测数量
            cv2.putText(
                img_with_boxes,
                f"Total: {len(detections)} detections",
                (min_x, min_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # 保存结果（使用5位数字格式化帧编号）
            cv2.imwrite(str(OUTPUT_DIR / f"frame_{idx:05d}.png"), img_with_boxes)

    print(f"处理完成！共处理 {idx + 1} 帧，结果保存在 {OUTPUT_DIR}")
