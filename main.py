import time

from muses_project.controller import ProjectController
from muses_project.graph_eeg import graph_eeg
import asyncio


if __name__ == "__main__":
    controller = ProjectController()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(controller.connect_to_hand())
    loop.run_until_complete(controller.move_hand("open"))
    loop.run_until_complete(controller.move_hand("close"))
    controller.connect_to_stream()
    try:
        graph_eeg(controller)
    except KeyboardInterrupt:
        print("Interrupted Manually. Exiting...")
    finally:
        controller.disconnect()