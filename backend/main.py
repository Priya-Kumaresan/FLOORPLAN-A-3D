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
        img_bytes = await file.read()

        # 1. segmentation
        wall_mask = predict_wall_mask(img_bytes)

        # 2. wall line extraction
        wall_lines = extract_wall_lines(wall_mask)

        # 3. 3D mesh
        wall_mesh = extrude_walls(wall_lines)

        # 4. furniture (optional)
        full_mesh = add_basic_furniture(wall_mesh)

        # 5. export GLB
        out_path = export_to_glb(full_mesh)

        return FileResponse(out_path, media_type="model/gltf-binary")

    except Exception as e:
        print("[/convert] ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
