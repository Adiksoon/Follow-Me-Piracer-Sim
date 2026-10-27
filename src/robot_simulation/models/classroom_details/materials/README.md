# CP46 floor appearance asset

The production classroom world uses this matte terrazzo albedo for its existing
floor visual. The floor mesh, collision, pose, friction and physics are unchanged.
The existing `classroom_details` model directory is only the resource container;
no additional model is spawned for the floor.

- Material: `PiRacer/CP46FloorTerrazzo` in `scripts/cp46_floor.material`.
- Texture: `textures/cp46_floor_terrazzo.png`, original opaque RGB PNG,
  1254 x 1254 pixels, 3,575,528 bytes.
- SHA-256: `971b39e6e4c6ed7df937c30dec28c7a42ef93ab4e78302a742af7a2f4ac6fd40`.
- Original project asset generated with the built-in OpenAI image generation tool
  on 2026-10-27 (one generation, no subsequent raster edits or upscaling).
  No external photograph or third-party texture was copied. This generated asset
  is separate from the PiRacer mesh and suitcase asset licenses.
- The prompt requested 2048 x 2048; the tool returned 1254 x 1254. The delivered
  resolution is recorded here rather than treating an upsample as extra detail.
- Intended mapping: one nonrepeating atlas over the existing 15.4 x 15.4 m box
  top face, using the box's existing UVs. At that mapping the sample spacing is
  approximately 12.28 mm/texel. Aggregate dimensions are approximate, not calibrated.
  The default box UV mapping and camera appearance still require runtime checking.
- Clamp addressing avoids tiled repetition; trilinear minification/mipmap filtering
  is requested to reduce aliasing. It does not prove camera-level visual quality.
- Nonzero ambient/diffuse and zero specular preserve a matte surface. There is no
  normal, displacement or height map and no geometric surface roughness change.
- Existing recursive package installation includes both resource directories.
  The world explicitly references the script and texture directory through
  `model://classroom_details/...`; the existing launch provides the models path.

Improved VO tracking has not been demonstrated by this offline change.

## Exact generation prompt

Create exactly one 2048 by 2048 pixel square, photorealistic PBR-style diffuse/albedo texture atlas for the entire 15.4 metre by 15.4 metre floor of a realistic school laboratory. It must be an orthographic straight-down flat surface, edge-to-edge, without perspective, borders, objects, text, labels, logos or watermark. This is a texture image, not a picture of a room.

The material is a natural matte warm grey-beige terrazzo or coarse aggregate floor, with irregular angular mineral aggregate pieces in restrained taupe, sandy beige, warm grey and occasional muted darker grey. Aggregate pieces have varied shapes, sizes and spacing, with the visually important pieces approximately 8 to 16 centimetres wide in this full-floor atlas (roughly 11 to 21 pixels in a 2048 pixel image). These larger irregular details must be distributed across the whole image, neither sparse nor identical. Include gently varying neutral base tone and believable natural variation; avoid fine speckle as the main source of detail. The desired appearance is realistic, understated and clean enough for an engineering laboratory, with moderate local tonal contrast so a 640 by 360 pixel RGB camera can track distinctive details from 1 to 3 metres. No pure black or stark white chips, no checkerboards, no geometric grids, no repeating tiles or periodic motifs, no glossy reflections, relief, cracks, damage or obstacles. Use even neutral illumination with no baked shadows, highlights or directional light. One nonrepeating atlas representing the whole floor, not a repeated small seamless tile. Do not add any camera framing or scene elements. Opaque background.
