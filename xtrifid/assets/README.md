# Human geometry and pose weights

The skin is adapted from the MakeHuman hm08 base mesh distributed with MakeHuman 1.3.0, commit `1f508f6083b2f823dab15de924b3bde72e08d77c`. The original mesh credits Data Collection AB, Joel Palmius and Jonas Hauquier (2020). Its header identifies the asset as CC0 1.0 Universal. The pose volume derives from the default weights by the same authors (2021), also designated CC0. [Original base mesh](https://github.com/makehumancommunity/makehuman/blob/1f508f6083b2f823dab15de924b3bde72e08d77c/makehuman/data/3dobjs/base.obj) · [Original weights](https://github.com/makehumancommunity/makehuman/blob/1f508f6083b2f823dab15de924b3bde72e08d77c/makehuman/data/rigs/default_weights.mhw) · [CC0 asset license](https://github.com/makehumancommunity/makehuman/blob/1f508f6083b2f823dab15de924b3bde72e08d77c/LICENSE.ASSETS.md)

Xtrifid uses the `body` face group, converts its coordinates to unit stature and fits closed, untextured ocular surfaces to the left and right orbital helpers. The limbs, fingers, face, eyelids and ears share a connected skin. Game meshes use attribute-aware simplification with extra resolution at the face. The reduced render mesh keeps surviving original quads. Medium Render retains the authored control cage. High Render applies one Catmull–Clark subdivision step. The result contains a neutral base anatomy with geometry for the iris and pupil. The separate pose file stores the influence of the upper and lower limbs in rest space.

## Rebuilding the assets

The shipped meshes are ready to load. Only asset maintainers need the source files and the meshoptimizer v1.3 simplifier. The bake uses its C ABI from commit `9e1f07b159d3cb777f1c67ed31fc11fd117986f4`. [meshoptimizer v1.3 header](https://github.com/zeux/meshoptimizer/blob/9e1f07b159d3cb777f1c67ed31fc11fd117986f4/src/meshoptimizer.h)

With the checkout at `/work/meshoptimizer`, run these commands in a development container with Python, NumPy and a C++ compiler. `base.obj` and `default_weights.mhw` are the pinned MakeHuman inputs linked above.

```sh
g++ -std=c++17 -O2 -shared -fPIC -o /work/meshopt.so \
    /work/meshoptimizer/src/simplifier.cpp /work/meshoptimizer/src/allocator.cpp
python3 tools/bake_human.py /work/base.obj /work/meshopt.so xtrifid/assets
python3 tools/bake_human_rig.py /work/base.obj /work/default_weights.mhw xtrifid/assets/human-rig.npz
python3 -m unittest discover -s tests -p test_face.py
```

The SHA-256 digest of `base.obj` is `8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c`. The digest of `default_weights.mhw` is `0f3641d651ae3d00ad6b4ccee43142edb109d3bd909d27d9e4139ef1beed8625`.

## Adult physique morphs

`human-male.npz` and `human-female.npz` store geometric changes to the same hm08 skin and eyes. For each adult physique, Xtrifid averages the three MakeHuman `*-young.target` ancestry macro targets with equal weights. This uses MakeHuman's young-adult age setting and its average muscle and weight settings. The original assets credit Manuel Bastioni (2014) and the 2020 copyright holders Data Collection AB, Joel Palmius and Jonas Hauquier. Their headers release them under CC0 1.0. [Pinned MakeHuman macro targets](https://github.com/makehumancommunity/makehuman/tree/1f508f6083b2f823dab15de924b3bde72e08d77c/makehuman/data/targets/macrodetails) · [MakeHuman asset license](https://github.com/makehumancommunity/makehuman/blob/1f508f6083b2f823dab15de924b3bde72e08d77c/LICENSE.ASSETS.md)

Vertex offsets follow the same original vertices, simplification and subdivision as each neutral LOD. The bake fits the eyes to the morphed orbital helpers, carries the joint pivots and palm landmarks to the new anatomy, and measures the ground position for each pose. Runtime loads one signed 16-bit offset triplet per vertex for the selected physique and LOD. The neutral figure uses the original arrays directly.

To reproduce both files, place the six `african|asian|caucasian` × `male|female` young-adult targets in `/work/macrodetails` and run the following from the repository root after baking the base meshes and pose weights. The tool checks the source mesh and all six target digests before accepting their values.

```sh
python3 -m tools.bake_physiques /work/base.obj /work/macrodetails xtrifid/assets
python3 -m unittest discover -s tests -p test_physique.py
```
