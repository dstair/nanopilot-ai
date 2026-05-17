"""
Subprocess camera worker for gz-transport.

Runs in a separate process to avoid libprotobuf conflicts with TensorFlow.
Captures frames from Gazebo via gz-transport and writes them to shared memory
for the main process to read.

Shared memory layout:
    [1 byte sequence counter][H * W * 3 bytes frame data (RGB uint8)]

Usage (launched automatically by drone_gym.py):
    python gz_camera_worker.py <topic> <image_w> <image_h> <shm_name>
"""

import sys
import time
import logging

import cv2
import numpy as np
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [gz_camera_worker] %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <topic> <image_w> <image_h> <shm_name>",
              file=sys.stderr)
        sys.exit(1)

    topic = sys.argv[1]
    image_w = int(sys.argv[2])
    image_h = int(sys.argv[3])
    shm_name = sys.argv[4]

    frame_size = image_h * image_w * 3
    seq = 0

    # Attach to shared memory created by the parent process
    shm = SharedMemory(name=shm_name, create=False)
    # Parent owns unlinking. Avoid child resource_tracker racing parent cleanup.
    resource_tracker.unregister(shm._name, "shared_memory")
    logger.info("Attached to shared memory '%s' (%d bytes)", shm_name, shm.size)

    frame_count = [0]

    def on_image(msg):
        nonlocal seq
        try:
            frame_count[0] += 1
            if frame_count[0] % 30 == 1:
                logger.info("Receiving frames (count=%d)", frame_count[0])

            pixel_format = getattr(msg, 'pixel_format_type', 3)
            if pixel_format == 1:  # L_INT8 grayscale
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif pixel_format == 4:  # RGBA_INT8
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 4)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif pixel_format == 5:  # BGRA_INT8
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 4)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            else:  # RGB_INT8 (3) or unknown
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)

            resized = cv2.resize(arr, (image_w, image_h))
            frame_bytes = resized.tobytes()

            # Write frame then bump sequence counter
            shm.buf[1:1 + frame_size] = frame_bytes
            seq = (seq + 1) % 256
            shm.buf[0] = seq
        except Exception as e:
            logger.warning("Error processing gz image: %s", e)

    node = Node()
    ok = node.subscribe(GzImage, topic, on_image)
    if ok:
        logger.info("Subscribed to %s", topic)
    else:
        logger.error("Failed to subscribe to %s", topic)
        shm.close()
        sys.exit(1)

    # Block until parent kills us
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        logger.info("Worker exiting")


if __name__ == "__main__":
    main()
