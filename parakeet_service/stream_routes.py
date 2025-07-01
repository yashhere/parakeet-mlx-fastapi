from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
from parakeet_service.config import logger

router = APIRouter()


@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    """
    WebSocket endpoint for real-time speech recognition using parakeet-mlx streaming.
    Accepts 16kHz mono PCM audio frames and returns transcription results.
    """
    await ws.accept()

    # Get the model from the app state
    model = ws.app.state.asr_model

    try:
        # Use parakeet-mlx's streaming transcription
        with model.transcribe_stream(
            context_size=(256, 256),  # (left_context, right_context) frames
            depth=1,  # Number of encoder layers that preserve exact computation
            keep_original_attention=False,  # Use local attention for streaming
        ) as transcriber:

            logger.info("WebSocket streaming session started")

            while True:
                try:
                    # Receive audio frame from client
                    frame_bytes = await ws.receive_bytes()

                    # Convert bytes to float32 audio data
                    pcm_data = (
                        np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    # Add audio to the streaming transcriber
                    transcriber.add_audio(pcm_data)

                    # Get current transcription result
                    result = transcriber.result

                    # Send transcription back to client
                    if result and result.text:
                        await ws.send_json(
                            {
                                "text": result.text,
                                "is_final": False,  # Streaming results are typically partial
                            }
                        )

                except WebSocketDisconnect:
                    logger.info("WebSocket client disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}")
                    await ws.send_json({"error": str(e)})
                    break

    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
        try:
            await ws.send_json({"error": f"Streaming failed: {str(e)}"})
        except:  # noqa
            pass  # Connection might already be closed
