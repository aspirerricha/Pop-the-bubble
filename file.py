
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

from mediapipe.tasks.python import BaseOptions, vision

# ==========================================
# 1. GAME SETTINGS
# ==========================================

WIDTH = 960
HEIGHT = 540
GAME_DURATION = 60
PINCH_THRESHOLD = 0.05
OPEN_PALM_HOLD_TIME = 0.6  # Open palm must be detected this long to pause

# Theme colors (BGR)
BG = (20, 12, 35)
PURPLE = (220, 70, 180)
NEON = (255, 100, 255)
CYAN = (255, 220, 70)
WHITE = (255, 255, 255)
GREEN = (80, 255, 120)
RED = (80, 80, 255)
PANEL = (45, 25, 65)

score = 0
lives = 5
bubbles = []
particles = []

game_state = "MENU"

start_time = 0
last_spawn = 0
last_cast = 0

# Pause tracking
paused_at = None
total_paused_time = 0
open_palm_start = None

pinch_previous = False
open_palm_previous = False


# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2
    )


def finger_up(hand, tip, joint):
    return hand[tip].y < hand[joint].y


# OPEN PALM = PAUSE

def is_open_palm(hand):

    index = finger_up(hand, 8, 6)
    middle = finger_up(hand, 12, 10)
    ring = finger_up(hand, 16, 14)
    pinky = finger_up(hand, 20, 18)

    return (
        index and
        middle and
        ring and
        pinky
    )


# YOLO / ROCK SIGN = SHIELD

def is_yolo(hand):

    index = finger_up(hand, 8, 6)
    middle = finger_up(hand, 12, 10)
    ring = finger_up(hand, 16, 14)
    pinky = finger_up(hand, 20, 18)

    return (
        index and
        not middle and
        not ring and
        pinky
    )


def create_bubble():

    bubbles.append({
        "x": random.randint(60, WIDTH - 60),
        "y": HEIGHT + 30,
        "r": random.randint(20, 35),
        "speed": random.uniform(3.0, 5.0)
    })


def create_particles(x, y):

    for _ in range(18):

        particles.append({
            "x": x,
            "y": y,
            "dx": random.uniform(-5, 5),
            "dy": random.uniform(-5, 5),
            "life": 20
        })


def reset_game():

    global score, lives, bubbles, particles
    global start_time, last_spawn, last_cast
    global game_state, pinch_previous
    global open_palm_previous
    global paused_at, total_paused_time, open_palm_start

    score = 0
    lives = 5

    bubbles = []
    particles = []

    start_time = time.time()
    last_spawn = start_time
    last_cast = 0

    paused_at = None
    total_paused_time = 0
    open_palm_start = None

    pinch_previous = False
    open_palm_previous = False

    game_state = "PLAYING"


# ==========================================
# 4. UI DRAWING FUNCTIONS
# ==========================================

def draw_panel(frame, x, y, w, h, color=PANEL):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        color,
        -1,
        cv2.LINE_AA
    )

    cv2.addWeighted(
        overlay, 0.75,
        frame, 0.25,
        0, frame
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        PURPLE,
        2,
        cv2.LINE_AA
    )


def draw_text(frame, text, position, size=0.7,
              color=WHITE, thickness=2):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_center_text(frame, text, y, size=1,
                     color=WHITE, thickness=2):

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(
        text, font, size, thickness
    )[0]

    x = (WIDTH - text_size[0]) // 2

    draw_text(
        frame,
        text,
        (x, y),
        size,
        color,
        thickness
    )


# ==========================================
# 5. BUBBLE AND PARTICLE DRAWING
# ==========================================

def draw_bubble(frame, bubble):

    x = int(bubble["x"])
    y = int(bubble["y"])
    r = bubble["r"]

    # Glow
    cv2.circle(
        frame, (x, y),
        r + 9, PURPLE, 2
    )

    cv2.circle(
        frame, (x, y),
        r + 4, NEON, 2
    )

    # Bubble
    cv2.circle(
        frame, (x, y),
        r, (255, 100, 230), 2
    )

    # Reflection
    cv2.circle(
        frame,
        (x - 8, y - 8),
        5,
        WHITE,
        -1
    )


def draw_particles(frame):

    for p in particles:

        cv2.circle(
            frame,
            (int(p["x"]), int(p["y"])),
            3,
            CYAN,
            -1
        )


# ==========================================
# 6. GAME HUD
# ==========================================

def draw_hud(frame, remaining):

    # SCORE
    draw_panel(frame, 20, 15, 180, 65)

    draw_text(
        frame, "SCORE",
        (38, 40), 0.5, CYAN
    )

    draw_text(
        frame, str(score),
        (38, 67), 0.8, WHITE
    )

    # LIVES
    draw_panel(frame, 215, 15, 180, 65)

    draw_text(
        frame, "LIVES",
        (235, 40), 0.5, CYAN
    )

    draw_text(
        frame, "♥ " * lives,
        (235, 67), 0.7, RED
    )

    # TIMER
    draw_panel(frame, 410, 15, 140, 65)

    draw_text(
        frame, "TIME",
        (430, 40), 0.5, CYAN
    )

    draw_text(
        frame, str(remaining) + "s",
        (430, 67), 0.8, WHITE
    )

    # TIMER PROGRESS BAR
    bar_width = int(
        200 * remaining / GAME_DURATION
    )

    cv2.rectangle(
        frame,
        (WIDTH - 225, 35),
        (WIDTH - 25, 48),
        (70, 50, 80),
        -1
    )

    cv2.rectangle(
        frame,
        (WIDTH - 225, 35),
        (WIDTH - 225 + bar_width, 48),
        CYAN if remaining > 15 else RED,
        -1
    )


# ==========================================
# 7. START SCREEN
# ==========================================

def draw_menu(frame):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (WIDTH, HEIGHT),
        BG,
        -1
    )

    cv2.addWeighted(
        overlay, 0.85,
        frame, 0.15,
        0, frame
    )

    draw_center_text(
        frame,
        "PINCH THE BUBBLE",
        150, 1.5, NEON, 3
    )

    draw_center_text(
        frame,
        "Your hands. Your magic. Your game.",
        195, 0.65, CYAN
    )

    draw_panel(frame, 230, 230, 500, 190)

    draw_center_text(
        frame, "HOW TO PLAY",
        270, 0.8, WHITE
    )

    draw_center_text(
        frame, "PINCH - POP BUBBLES",
        310, 0.6, NEON
    )

    draw_center_text(
        frame, "OPEN PALM - PAUSE",
        345, 0.6, CYAN
    )

    draw_center_text(
        frame, "YOLO SIGN - SHIELD",
        380, 0.6, GREEN
    )

    draw_center_text(
        frame, "PRESS SPACE TO START",
        480, 0.8, WHITE
    )

    draw_center_text(
        frame, "Press Q to quit",
        515, 0.45, CYAN
    )


# ==========================================
# 8. PAUSE SCREEN
# ==========================================

def draw_pause_screen(frame):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (WIDTH, HEIGHT),
        BG,
        -1
    )

    cv2.addWeighted(
        overlay, 0.7,
        frame, 0.3,
        0, frame
    )

    draw_panel(frame, 260, 170, 440, 200)

    draw_center_text(
        frame, "GAME PAUSED",
        235, 1.2, NEON
    )

    draw_center_text(
        frame, "Take a breather!",
        285, 0.7, WHITE
    )

    draw_center_text(
        frame, "PRESS SPACE TO RESUME",
        330, 0.65, CYAN
    )

    draw_center_text(
        frame, "Press R to restart",
        355, 0.5, WHITE
    )


# ==========================================
# 9. GAME OVER SCREEN
# ==========================================

def draw_game_over(frame):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (WIDTH, HEIGHT),
        BG,
        -1
    )

    cv2.addWeighted(
        overlay, 0.85,
        frame, 0.15,
        0, frame
    )

    draw_panel(frame, 230, 140, 500, 270)

    draw_center_text(
        frame, "GAME OVER!",
        210, 1.4, RED, 3
    )

    draw_center_text(
        frame, "FINAL SCORE",
        270, 0.7, CYAN
    )

    draw_center_text(
        frame, str(score),
        330, 1.5, WHITE, 3
    )

    draw_center_text(
        frame, "PRESS R TO PLAY AGAIN",
        375, 0.65, NEON
    )

    draw_center_text(
        frame, "Press Q to quit",
        450, 0.5, WHITE
    )


# ==========================================
# 10. MAIN GAME LOOP
# ==========================================

with vision.HandLandmarker.create_from_options(options) as landmarker:

    last_timestamp = 0

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT)
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = max(
            last_timestamp + 1,
            int(time.perf_counter() * 1000)
        )

        last_timestamp = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        current_time = time.time()

        cursor = None
        shield_active = False

        # ==================================
        # 11. MENU
        # ==================================

        if game_state == "MENU":

            draw_menu(frame)

        # ==================================
        # 12. HAND GESTURE RECOGNITION
        # ==================================

        if (
            result.hand_landmarks
            and game_state == "PLAYING"
        ):

            hand = result.hand_landmarks[0]

            tip = hand[8]

            cursor = (
                int(tip.x * WIDTH),
                int(tip.y * HEIGHT)
            )

            # Draw hand landmarks
            for lm in hand:

                x = int(lm.x * WIDTH)
                y = int(lm.y * HEIGHT)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    GREEN,
                    -1
                )

            # Detect gestures
            pinch_distance = distance(
                hand[4],
                hand[8]
            )

            pinch = (
                pinch_distance < PINCH_THRESHOLD
            )

            open_palm = is_open_palm(hand)

            yolo = is_yolo(hand)

            # OPEN PALM = PAUSE
            # The palm must stay detected continuously for a short time.
            # This prevents accidental pauses caused by brief detection errors.
            if open_palm:
                if open_palm_start is None:
                    open_palm_start = current_time

                elif current_time - open_palm_start >= OPEN_PALM_HOLD_TIME:
                    game_state = "PAUSED"
                    paused_at = current_time
                    pinch_previous = False
                    open_palm_previous = True
                    open_palm_start = None
            else:
                open_palm_start = None
                open_palm_previous = False

            # YOLO = SHIELD
            shield_active = yolo and not pinch

            # PINCH = POP
            if (
                pinch
                and not pinch_previous
                and current_time - last_cast > 0.35
                and cursor is not None
            ):

                last_cast = current_time

                create_particles(
                    cursor[0],
                    cursor[1]
                )

                remaining_bubbles = []

                for bubble in bubbles:

                    d = math.hypot(
                        bubble["x"] - cursor[0],
                        bubble["y"] - cursor[1]
                    )

                    if d < bubble["r"] + 65:

                        score += 10

                        create_particles(
                            bubble["x"],
                            bubble["y"]
                        )

                    else:
                        remaining_bubbles.append(bubble)

                bubbles = remaining_bubbles

            pinch_previous = pinch

            # SHIELD VISUAL
            if shield_active:

                cv2.circle(
                    frame,
                    cursor,
                    85,
                    CYAN,
                    3
                )

                draw_text(
                    frame,
                    "SHIELD ACTIVE!",
                    (WIDTH - 250, 100),
                    0.6,
                    GREEN
                )

        # ==================================
        # 13. GAME TIMER
        # ==================================

        if game_state == "PLAYING":

            elapsed = (
                current_time
                - start_time
                - total_paused_time
            )

        elif game_state == "PAUSED":

            # Freeze timer at the exact pause moment
            elapsed = (
                paused_at
                - start_time
                - total_paused_time
            )

        else:

            elapsed = 0

        if game_state in ["PLAYING", "PAUSED"]:

            remaining = max(
                0,
                GAME_DURATION - int(elapsed)
            )

        # ==================================
        # 14. GAME LOGIC
        # ==================================

        if game_state == "PLAYING":

            if remaining <= 0 or lives <= 0:

                game_state = "GAME_OVER"

            else:

                # Spawn bubbles
                if current_time - last_spawn > 0.7:

                    create_bubble()
                    last_spawn = current_time

                # Move bubbles
                for bubble in bubbles:

                    if shield_active:

                        bubble["y"] -= (
                            bubble["speed"] * 2
                        )

                    else:

                        bubble["y"] -= bubble["speed"]

                # Remove escaped bubbles
                active_bubbles = []

                for bubble in bubbles:

                    if bubble["y"] < -40:

                        if not shield_active:
                            lives -= 1

                    else:
                        active_bubbles.append(bubble)

                bubbles = active_bubbles

                # Update particles
                for p in particles:

                    p["x"] += p["dx"]
                    p["y"] += p["dy"]
                    p["life"] -= 1

                particles = [
                    p for p in particles
                    if p["life"] > 0
                ]

                if lives <= 0:

                    game_state = "GAME_OVER"

        # ==================================
        # 15. DRAW GAME ELEMENTS
        # ==================================

        if game_state in ["PLAYING", "PAUSED"]:

            for bubble in bubbles:

                draw_bubble(frame, bubble)

            draw_particles(frame)

            if cursor is not None:

                cv2.circle(
                    frame,
                    cursor,
                    12,
                    CYAN,
                    2
                )

            draw_hud(frame, remaining)

            draw_text(
                frame,
                "PINCH: POP | OPEN PALM: PAUSE | YOLO: SHIELD",
                (20, HEIGHT - 20),
                0.48,
                WHITE
            )

        # ==================================
        # 16. PAUSE SCREEN
        # ==================================

        if game_state == "PAUSED":

            draw_pause_screen(frame)

        # ==================================
        # 17. GAME OVER SCREEN
        # ==================================

        if game_state == "GAME_OVER":

            draw_game_over(frame)

      

        cv2.imshow(
            "PINCH THE BUBBLE",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # QUIT
        if key == ord("q"):
            break

        # START GAME
        if (
            key == ord(" ")
            and game_state == "MENU"
        ):

            reset_game()

        # RESUME GAME
        elif (
            key == ord(" ")
            and game_state == "PAUSED"
        ):

            # Add paused duration to total
            total_paused_time += (
                time.time() - paused_at
            )

            paused_at = None

            game_state = "PLAYING"

            pinch_previous = False
            open_palm_start = None

            # Prevent immediate re-pause
            # if the open palm is still visible
            open_palm_previous = True

        # RESTART GAME
        if (
            key == ord("r")
            and game_state in ["GAME_OVER", "PAUSED"]
        ):

            reset_game()




cap.release()
cv2.destroyAllWindows()
