"""List cylindrical faces of the 38172000_01 STEP (read-only) to count true CAD bends."""
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
r = STEPControl_Reader(); r.ReadFile("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp"); r.TransferRoots(); s = r.OneShape()
ex = TopExp_Explorer(s, TopAbs_FACE); rows = []
while ex.More():
    f = TopoDS.Face_s(ex.Current()); a = BRepAdaptor_Surface(f)
    if a.GetType() == GeomAbs_Cylinder:
        cy = a.Cylinder(); ax = cy.Axis(); d = ax.Direction(); l = ax.Location()
        b = Bnd_Box(); BRepBndLib.Add_s(f, b); x0, y0, z0, x1, y1, z1 = b.Get()
        g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g)
        u0, u1 = a.FirstUParameter(), a.LastUParameter()
        rows.append((round(cy.Radius(), 2), tuple(round(v, 3) for v in (d.X(), d.Y(), d.Z())), (round(x0,1), round(y0,1), round(z0,1)), (round(x1,1), round(y1,1), round(z1,1)), round(g.Mass(), 1), round((u1-u0)*57.2958, 1)))
    ex.Next()
rows.sort(key=lambda r: (r[1], r[2]))
for r_ in rows: print(r_)
print(len(rows))
