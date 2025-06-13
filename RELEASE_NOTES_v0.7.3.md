# MuJoCo MCP v0.7.3 Release Notes

**Release Date**: 2025-06-13  
**Version**: 0.7.3  
**Type**: Feature Release

## 🎉 New Features

### 📸 Render Capture Capability

Added the ability to capture rendered images from the MuJoCo viewer, enabling:
- **Screenshot capture**: Get the current view as a PNG image
- **Configurable resolution**: Support for different image sizes (within framebuffer limits)
- **Base64 encoding**: Images returned as base64-encoded strings for easy transfer
- **MCP tool integration**: New `capture_render` tool available through MCP interface

## 🔧 Technical Details

### Implementation

1. **Viewer Server Enhancement** (`mujoco_viewer_server.py`):
   - Added `capture_render` command handler
   - Uses MuJoCo's `Renderer` class for offscreen rendering
   - Converts rendered pixels to PNG format using PIL
   - Returns base64-encoded image data

2. **Viewer Client Update** (`viewer_client.py`):
   - Added `capture_render()` method
   - Supports width/height parameters
   - Maintains backward compatibility

3. **Remote Server Integration** (`remote_server.py`):
   - New MCP tool: `capture_render`
   - Parameters: `model_id`, `width` (default: 640), `height` (default: 480)
   - Returns image data with metadata

### Usage Example

```python
# Capture current view
result = server.call_tool("capture_render", {
    "model_id": model_id,
    "width": 640,
    "height": 480
})

if result.get("success"):
    image_data = result.get("image_data")  # Base64 encoded PNG
    # Save to file
    with open("screenshot.png", "wb") as f:
        f.write(base64.b64decode(image_data))
```

## ⚠️ Known Limitations

1. **Framebuffer Size**: Image dimensions limited by model's offscreen framebuffer size
   - Default: 640x480
   - Larger sizes require XML configuration:
   ```xml
   <visual>
     <global offwidth="1920" offheight="1080"/>
   </visual>
   ```

2. **macOS Requirement**: Must run viewer server with `mjpython` on macOS

3. **Single Viewer Mode**: Render capture works with the current single-viewer architecture

## 🧪 Testing

- ✅ Basic render capture (pendulum, cart-pole)
- ✅ Different resolutions (320x240, 640x480)
- ✅ Image format verification (PNG)
- ✅ Base64 encoding/decoding
- ⚠️ Large resolution tests show framebuffer limits

## 🔄 Backward Compatibility

- All existing tools and features remain unchanged
- New tool is optional and doesn't affect existing workflows
- Previous versions can ignore the new tool

## 📝 Migration Guide

No migration needed. To use the new feature:

1. Update to v0.7.3
2. Ensure viewer server is running (with mjpython on macOS)
3. Use the new `capture_render` tool

## 🚀 Future Enhancements

Potential improvements for future releases:
- Video recording capability
- Multiple camera angles
- Depth/segmentation rendering
- Dynamic framebuffer resizing
- GIF animation export

## 📊 Performance Impact

- Minimal impact on simulation performance
- Render capture takes ~50-100ms depending on resolution
- No impact when feature is not used

## 🙏 Acknowledgments

Thanks to the MuJoCo team for the excellent rendering APIs that made this feature possible.

---

**Full Changelog**: v0.7.2...v0.7.3