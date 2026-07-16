"""WebSocket consumer for real-time progress updates."""

import json
import shutil
import asyncio
from pathlib import Path
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from asgiref.sync import sync_to_async
import logging
logger = logging.getLogger(__name__)

from .pipeline_runner import PipelineRunner


class ProcessingConsumer(AsyncWebsocketConsumer):
    """Handles WebSocket connections for pipeline processing."""

    async def connect(self):
        """Accept WebSocket connection."""
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        await self.accept()

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        pass

    async def receive(self, text_data):
        """Handle incoming WebSocket message."""
        data = json.loads(text_data)

        if data.get('action') == 'start_processing':
            # Start processing in background
            asyncio.create_task(self.process_image())

    async def process_image(self):
        """Process the uploaded image through the pipeline."""
        try:
            # Find uploaded file
            upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
            upload_files = list(upload_dir.glob(f"{self.session_id}.*"))

            if not upload_files:
                await self.send_error("Uploaded file not found")
                return

            upload_path = str(upload_files[0])

            # Progress callback
            async def progress_callback(percentage: int, message: str):
                await self.send_progress(percentage, message)

            # Run pipeline in thread pool
            runner = PipelineRunner()

            # Get event loop BEFORE defining the callback
            loop = asyncio.get_event_loop()
            progress_queue = asyncio.Queue()

            def sync_progress_callback(percentage, message):
                # Put progress updates in queue for async processing
                try:
                    progress_queue.put_nowait((percentage, message))
                except asyncio.QueueFull:
                    pass  # Skip if queue is full

            # Start pipeline in background (run_in_executor returns Future, not coroutine)
            pipeline_future = loop.run_in_executor(
                None,
                lambda: runner.run_pipeline(upload_path, sync_progress_callback)
            )

            # Process progress updates while pipeline runs
            success, csv_path, img_path = await self._handle_long_pipeline(
                pipeline_future, progress_queue
            )
            
            
            logger.info(f"Success: {success}, csv_path: {csv_path}, img_path: {img_path}")
            if success and csv_path and img_path:
                # Copy results to media directory
                results_dir = Path(settings.MEDIA_ROOT) / 'results'
                results_dir.mkdir(parents=True, exist_ok=True)

                csv_dest = results_dir / f"{self.session_id}.csv"
                img_dest = results_dir / f"{self.session_id}.png"

                await sync_to_async(shutil.copy2)(csv_path, csv_dest)
                await sync_to_async(shutil.copy2)(img_path, img_dest)

                # Send completion message
                await self.send_complete()
            else:
                await self.send_error("Pipeline processing failed")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in process_image: {e}")
            print(f"Full traceback:\n{error_details}")
            await self.send_error(f"Error: {str(e)}")

    async def _handle_long_pipeline(self, pipeline_future, progress_queue):
        """
        Handle long-running pipeline with progress updates.
        Returns (success, csv_path, img_path) when complete.
        """
        try:
            while not pipeline_future.done():
                try:
                    # Wait for progress update or timeout
                    percentage, message = await asyncio.wait_for(
                        progress_queue.get(), timeout=5.0
                    )
                    await self.send_progress(percentage, message)

                    # Send periodic heartbeat to keep WebSocket alive
                    if percentage % 10 == 0:  # Every 10%
                        await self.send_heartbeat()

                except asyncio.TimeoutError:
                    # No progress update in 5 seconds, send heartbeat
                    await self.send_heartbeat()
                    continue
                except Exception as e:
                    print(f"Error processing progress: {e}")
                    continue

            # Pipeline completed, get results
            return await pipeline_future

        except Exception as e:
            print(f"Error in _handle_long_pipeline: {e}")
            pipeline_future.cancel()
            return False, None, None

    async def send_heartbeat(self):
        """Send heartbeat to keep WebSocket connection alive."""
        try:
            await self.send(text_data=json.dumps({
                'type': 'heartbeat',
                'timestamp': asyncio.get_event_loop().time()
            }))
        except Exception as e:
            print(f"Error sending heartbeat: {e}")

    async def send_progress(self, percentage: int, message: str):
        """Send progress update to client."""
        try:
            await self.send(text_data=json.dumps({
                'type': 'progress',
                'percentage': percentage,
                'message': message
            }))
        except Exception as e:
            print(f"Error sending progress: {e}")

    async def send_complete(self):
        """Send completion message to client."""
        await self.send(text_data=json.dumps({
            'type': 'complete',
            'session_id': self.session_id
        }))

    async def send_error(self, message: str):
        """Send error message to client."""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message
        }))
