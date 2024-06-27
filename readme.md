# Hiperwall Controller Multimodal
A Multimodal  User Interface for HiperController Software
## Description
Hiperwall Controller Multimodal integrates hand tracking using a webcam and voice recognition to control various mouse actions. By detecting specific hand gestures and interpreting voice commands, users can interact with their computer in a more intuitive and natural way.
## Getting Started
### Dependencies
Before you begin, ensure you have met the following requirements:
- Python 3.7 or higher
- OS: Windows 10 or higher, macOS 14, Linux
### Installing
1. Clone the repository:
```sh
git clone https://github.com/sion99/multimodal.git
cd multimodal
```
2. Create and activate a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
```
3. Install the required libraries:
```sh
pip install -r requirements.txt
```
### Executing Program
To run the program, follow these steps:

1. Start the Python script
```sh
python main.py
```
2. The webcam will activate, and the program will begin listening for voice commands and tracking hand gestures.
3. Use the following voice commands to control the mouse:
    - "two": Double-click
    - "left": Left-click
    - "right": Right-click
    - "up": Scroll up
    - "down": Scroll down
4. Press ESC to exit the program.
