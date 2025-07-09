# Muse S Project

This project allows control over a prosthetic hand from the organization Haifa3D using a Muse S EEG headband. The
classification of movements is done via a crude algorithm that should be tuned based on the user.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/afifisuha/muses_project.git
   ```
2. Navigate to the project directory:
   ```bash
    cd muses_project
   ```
3. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Turn on and wear the Muse S headband.
2. Turn on the prosthetic hand.
3. Run the main script:
   ```bash
   python main.py
   ```
4. After a short delay, you should see graphs on your screen of the information provided by the Muse S headband.
5. Blink hard to turn on controlling the hand.
6. Turning your head gently up, down, left, and right will control the hand's movement.

## Notes

Special thanks to:

The Haifa3D organization for providing the prosthetic hand.

The Muse S team for creating the EEG headband.

The muselsl team for the [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) library, which is used to read the EEG data from the Muse S headband.

Liad Olier and Roee Savion, who created the [Mindrove_armband_EMG_classifier](https://github.com/liadolier99/Mindrove_armband_EMG_classifier)
repository, which control of the hand is based on.