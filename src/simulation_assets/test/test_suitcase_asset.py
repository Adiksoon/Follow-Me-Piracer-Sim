import hashlib
from pathlib import Path


def test_suitcase_mesh_matches_approved_geometry():
    mesh = Path(__file__).parents[1] / "assets/hard_trolley_red/meshes/suitcase.dae"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == (
        "e4d75b678e566289f15d58fdc7216c93f2199a34bb82377bd785d1e94a14f1ca"
    )
