# Geometry is built as meshes (SDF + manifold3d), not build123d B-rep

Every other project here is build123d B-rep built with cadgen. This one isn't: the Lattice's look depends on Struts that blend smoothly into thickened Nodes, which B-rep fillets can't do over hundreds of intersecting tubes, and the Stretch prototype starts from the Original's mesh, which has no B-rep at all. So the Lattice, Rails and Side guides are a signed distance field (Struts blended with a log-sum-exp smooth minimum) turned into a mesh with marching cubes, and the exact parts (Seam band, Pin bosses, holes, Pins) are added with manifold3d booleans. The Fit coupon uses the same joint code as the halves, so what it tests is what the halves get. cadgen is used only to view the meshes.

## Consequences

- No `.step` export: a mesh never becomes B-rep. `exports/` holds the plate 3MFs only.
- Measurements are on triangles (check.py), accurate to about the 0.4 mm field spacing for the Lattice and exact for the joint.
