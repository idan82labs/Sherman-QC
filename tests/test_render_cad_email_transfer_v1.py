"""Authored display contract only: no CAD/native imports or corpus inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.render_cad_email_transfer_v1 import pinned_json, render


class RenderContract(unittest.TestCase):
    def test_pin_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"e.json"; p.write_text("{}")
            with self.assertRaisesRegex(ValueError,"PIN_MISMATCH"): pinned_json(p,"0"*64)
            link=Path(folder)/"link.json"; link.symlink_to(p)
            with self.assertRaisesRegex(ValueError,"BUDGET_OR_TYPE"): pinned_json(link,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_neutral_and_overlay_retain_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); e=root/"e.json"; p=root/"p.json"
            e.write_text(json.dumps({"input_sha256":"a"*64,"faces":[{"id":1,"surface_class":"Cylinder","area_centroid":[.3,.3,0],"display_mesh":{"vertices":[[0,0,0],[1,0,0],[0,1,0]],"triangles_one_based":[[1,2,3]]}}]}))
            ends=[[.1,.1,0],[.4,.1,0]]
            p.write_text(json.dumps({"input_sha256":"a"*64,"bend_axes":[{"canonical_line_start":ends[0],"canonical_line_end":ends[1]}]}))
            es=hashlib.sha256(e.read_bytes()).hexdigest(); ps=hashlib.sha256(p.read_bytes()).hexdigest()
            render(e,es,root/"neutral.png")
            neutral=json.loads((root/"neutral.json").read_text())
            self.assertFalse(neutral["prediction_access"]); self.assertEqual(neutral["marker_count"],0)
            render(e,es,root/"overlay.png",p,ps)
            overlay=json.loads((root/"overlay.json").read_text())
            self.assertEqual(overlay["marker_endpoints"],[ends]); self.assertTrue(overlay["prediction_access"])
            with self.assertRaisesRegex(ValueError,"TERMINAL"): render(e,es,root/"neutral.png")
            p.write_text(json.dumps({"bend_axes":[]}))
            with self.assertRaisesRegex(ValueError,"INPUT_MISMATCH"):
                render(e,es,root/"unbound.png",p,hashlib.sha256(p.read_bytes()).hexdigest())


if __name__=="__main__": unittest.main()
