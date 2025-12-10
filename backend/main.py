import os
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from floorplan_ai import predict_wall_mask
from postprocess import extract_wall_lines
from build_3d import extrude_walls, export_to_glb
from furniture import add_basic_furniture


app = FastAPI()

# Allow requests from your JS frontend on localhost:5500 etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        print(f"[/convert] Received file: {file.filename}, size: {file.size}")
        img_bytes = await file.read()
        print(f"[/convert] Image bytes length: {len(img_bytes)}")

        # 1. segmentation
        print("[/convert] Step 1: Running AI segmentation...")
        wall_mask = predict_wall_mask(img_bytes)
        print(f"[/convert] Mask shape: {wall_mask.shape}, dtype: {wall_mask.dtype}")
        print(f"[/convert] Mask stats: min={wall_mask.min()}, max={wall_mask.max()}, sum={wall_mask.sum()}, mean={wall_mask.mean():.3f}")

        # Save generated mask to data/generated_masks/
        mask_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated_masks")
        os.makedirs(mask_dir, exist_ok=True)
        print(f"[/convert] Mask directory: {mask_dir} (exists: {os.path.exists(mask_dir)})")
        
        # Generate mask filename from input filename
        input_filename = file.filename or "uploaded_image.jpg"
        # Clean filename (remove path separators if any)
        safe_filename = os.path.basename(input_filename)
        mask_filename = os.path.splitext(safe_filename)[0] + "_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # Save mask as image (0=black background, 255=white walls)
        mask_image = (wall_mask * 255).astype('uint8')
        
        # Check if mask is not empty
        if wall_mask.sum() == 0:
            print("[/convert] ⚠️  WARNING: Generated mask is empty (no walls detected)!")
        else:
            print(f"[/convert] Mask contains {wall_mask.sum()} wall pixels")
        
        success = cv2.imwrite(mask_path, mask_image)
        if success:
            print(f"[/convert] 💾 Saved generated mask to: {mask_path}")
            print(f"[/convert] Mask file size: {os.path.getsize(mask_path)} bytes")
        else:
            print(f"[/convert] ❌ ERROR: Failed to save mask to {mask_path}")

        # 2. wall line extraction
        print("[/convert] Step 2: Extracting wall lines...")
        wall_lines = extract_wall_lines(wall_mask)
        print(f"[/convert] Found {len(wall_lines)} wall line segments")

        # 3. 3D mesh
        print("[/convert] Step 3: Building 3D mesh...")
        wall_mesh = extrude_walls(wall_lines)

        # 4. furniture (optional)
        print("[/convert] Step 4: Adding furniture...")
        full_mesh = add_basic_furniture(wall_mesh)

        # 5. export GLB
        print("[/convert] Step 5: Exporting GLB...")
        out_path = export_to_glb(full_mesh)
        print(f"[/convert] ✅ Success! GLB saved to: {out_path}")

        return FileResponse(out_path, media_type="model/gltf-binary")

    except Exception as e:
        print("[/convert] ❌ ERROR:", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
