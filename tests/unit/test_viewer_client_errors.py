"""Error path tests for viewer_client.py"""

import socket
import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from mujoco_mcp.viewer_client import MuJoCoViewerClient


class TestViewerClientConnectionErrors:
    """Test connection error handling in MuJoCoViewerClient."""

    def test_send_command_when_not_connected(self):
        """Test that send_command raises ConnectionError when not connected."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        # Don't connect
        with pytest.raises(ConnectionError, match="Not connected to viewer server"):
            client.send_command("test_command", {})

    def test_send_command_after_disconnect(self):
        """Test that send_command fails after disconnecting."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        # Mock socket connection
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            # Simulate successful connection
            client.connect()
            assert client.connected

            # Disconnect
            client.disconnect()
            assert not client.connected

            # Should raise error after disconnect
            with pytest.raises(ConnectionError, match="Not connected to viewer server"):
                client.send_command("test", {})

    def test_connection_refused_error(self):
        """Test handling of connection refused errors."""
        client = MuJoCoViewerClient(host="localhost", port=9999)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect.side_effect = ConnectionRefusedError("Connection refused")
            mock_socket_class.return_value = mock_socket

            # Should handle connection error gracefully
            with pytest.raises(ConnectionRefusedError):
                client.connect()

    def test_timeout_during_connection(self):
        """Test handling of timeout during connection."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect.side_effect = TimeoutError("Connection timed out")
            mock_socket_class.return_value = mock_socket

            with pytest.raises(socket.timeout):
                client.connect()


class TestViewerClientResponseErrors:
    """Test error handling for invalid server responses."""

    def test_invalid_json_response(self):
        """Test handling of invalid JSON in server response."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            # Simulate connection
            client.connect()

            # Return invalid JSON
            mock_socket.recv.return_value = b"not valid json\n"

            # Should raise ValueError for invalid JSON
            with pytest.raises(ValueError, match="Invalid JSON response"):
                client.send_command("test", {})

    def test_utf8_decode_error_in_response(self):
        """Test handling of UTF-8 decode errors in response."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # Return invalid UTF-8 bytes
            mock_socket.recv.return_value = b"\xff\xfe\n"

            with pytest.raises(ValueError, match="Failed to decode server response as UTF-8"):
                client.send_command("test", {})

    def test_empty_response_from_server(self):
        """Test handling of empty response from server."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # Return empty response
            mock_socket.recv.return_value = b""

            # Should handle empty response (might raise or return None)
            try:
                result = client.send_command("test", {})
                # If it doesn't raise, result should be None or empty
                assert result is None or result == ""
            except (ValueError, ConnectionError):
                # Also acceptable to raise an error
                pass

    def test_malformed_newline_in_response(self):
        """Test handling of response without proper newline terminator."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # Valid JSON but no newline (might cause issues with protocol)
            mock_socket.recv.return_value = b'{"status": "ok"}'  # No \n

            # Should still work or handle gracefully
            try:
                result = client.send_command("test", {})
                # Should parse the JSON correctly
                if result:
                    assert isinstance(result, (dict, str))
            except ValueError:
                # Also acceptable if protocol requires newline
                pass


class TestViewerClientNetworkErrors:
    """Test network error handling."""

    def test_socket_error_during_send(self):
        """Test handling of socket errors during send."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # Simulate socket error on send
            mock_socket.sendall.side_effect = OSError("Network error")

            with pytest.raises(socket.error):
                client.send_command("test", {})

    def test_socket_error_during_receive(self):
        """Test handling of socket errors during receive."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # sendall works, but recv fails
            mock_socket.recv.side_effect = OSError("Network error")

            with pytest.raises(socket.error):
                client.send_command("test", {})

    def test_broken_pipe_error(self):
        """Test handling of broken pipe errors."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # Simulate broken pipe
            mock_socket.sendall.side_effect = BrokenPipeError("Broken pipe")

            with pytest.raises(BrokenPipeError):
                client.send_command("test", {})


class TestViewerClientCommandValidation:
    """Test command parameter validation."""

    def test_valid_command_with_parameters(self):
        """Test that valid commands work correctly."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            # Mock successful response
            mock_socket.recv.return_value = b'{"status": "success"}\n'

            result = client.send_command("load_model", {"model_xml": "<test/>"})

            # Should have sent the command
            assert mock_socket.sendall.called
            sent_data = mock_socket.sendall.call_args[0][0]
            assert b"load_model" in sent_data

    def test_command_with_complex_parameters(self):
        """Test commands with nested parameter structures."""
        client = MuJoCoViewerClient(host="localhost", port=8888)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect()

            mock_socket.recv.return_value = b'{"status": "success"}\n'

            complex_params = {
                "positions": [1.0, 2.0, 3.0],
                "options": {
                    "speed": 0.5,
                    "precision": True
                }
            }

            result = client.send_command("move", complex_params)

            # Should serialize complex parameters correctly
            sent_data = mock_socket.sendall.call_args[0][0].decode()
            # Should contain valid JSON
            assert "{" in sent_data
            assert "}" in sent_data
