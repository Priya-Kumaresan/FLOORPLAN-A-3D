async function upload() {
  try {
    const fileInput = document.getElementById("inputImage").files[0];
    if (!fileInput) {
      alert("Please choose a floor plan image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput);

    const response = await fetch("http://127.0.0.1:8000/convert", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      alert("Error from backend:\n" + text);
      return;
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    loadModel(url);
  } catch (err) {
    alert("Error uploading file: " + err);
    console.error(err);
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
