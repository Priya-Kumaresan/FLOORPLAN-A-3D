async function upload() {
  try {
    const fileInput = document.getElementById("inputImage").files[0];
    if (!fileInput) {
      alert("Please choose a floor plan image first.");
      return;
    }

    const convertBtn = document.getElementById("convertBtn");
    const originalText = convertBtn.textContent;
    convertBtn.disabled = true;
    convertBtn.textContent = "Converting...";
    
    const viewer = document.getElementById("viewer");
    viewer.innerHTML = "<p style='padding: 20px; color: #666;'>Processing floor plan... This may take a moment.</p>";

    console.log("Uploading file:", fileInput.name);

    const formData = new FormData();
    formData.append("file", fileInput);

    console.log("Sending request to http://127.0.0.1:8000/convert");
    const response = await fetch("http://127.0.0.1:8000/convert", {
      method: "POST",
      body: formData,
    });

    console.log("Response status:", response.status, response.statusText);

    if (!response.ok) {
      const text = await response.text();
      console.error("Backend error:", text);
      viewer.innerHTML = `<p style='padding: 20px; color: red;'>Error: ${text}</p>`;
      convertBtn.disabled = false;
      convertBtn.textContent = originalText;
      alert("Error from backend:\n" + text);
      return;
    }

    console.log("Received GLB file, loading 3D model...");
    const blob = await response.blob();
    console.log("Blob size:", blob.size, "bytes");
    const url = URL.createObjectURL(blob);
    loadModel(url);
    
    convertBtn.disabled = false;
    convertBtn.textContent = originalText;
  } catch (err) {
    console.error("Upload error:", err);
    const viewer = document.getElementById("viewer");
    viewer.innerHTML = `<p style='padding: 20px; color: red;'>Error: ${err.message}</p>`;
    const convertBtn = document.getElementById("convertBtn");
    convertBtn.disabled = false;
    convertBtn.textContent = "Convert";
    alert("Error uploading file: " + err.message);
  }
}

function loadModel(url) {
  const container = document.getElementById("viewer");
  container.innerHTML = "";

  const width = container.clientWidth || window.innerWidth;
  const height = container.clientHeight || (window.innerHeight - 120);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f4f4);

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.set(0, -15, 10);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // LIGHTS
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);

  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(10, -10, 20);
  scene.add(dir);

  const loader = new THREE.GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      const model = gltf.scene;
      scene.add(model);

      // center model
      const box = new THREE.Box3().setFromObject(model);
      const center = new THREE.Vector3();
      box.getCenter(center);
      model.position.sub(center);

      camera.position.set(0, -20, 15);
      camera.lookAt(0, 0, 0);
    },
    undefined,
    (error) => {
      console.error("Error loading GLB:", error);
      alert("Error loading 3D model in browser.");
    }
  );

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}
