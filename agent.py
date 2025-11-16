import cv2
import threading
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google, noise_cancellation

# Import your custom modules
from Jarvis_prompts import instructions_prompt, Reply_prompts
from memory_loop import MemoryExtractor
from jarvis_reasoning import thinking_capability

load_dotenv()

# ----------- Fullscreen Video GUI -------------
def play_video_loop(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ Error: Could not open video file.")
        return

    # Create fullscreen window
    cv2.namedWindow("IRONMAN GUI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("IRONMAN GUI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Restart video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (1920, 1080))  # Adjust to your screen resolution
        cv2.imshow("IRONMAN GUI", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------- AI Assistant Logic -------------
class Assistant(Agent):
    def __init__(self, chat_ctx) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=instructions_prompt,
            llm=google.beta.realtime.RealtimeModel(voice="Charon"),
            tools=[thinking_capability]
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(preemptive_generation=True)
    current_ctx = session.history.items

    await session.start(
        room=ctx.room,
        agent=Assistant(chat_ctx=current_ctx),
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await session.generate_reply(instructions=Reply_prompts)
    conv_ctx = MemoryExtractor()
    await conv_ctx.run(current_ctx)

if __name__ == "__main__":
    # Run the video in a separate thread so AI runs simultaneously
    video_thread = threading.Thread(target=play_video_loop, args=("ironman_converted_reencoded.mp4",), daemon=True)
    video_thread.start()

    # Start the Jarvis AI backend
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
