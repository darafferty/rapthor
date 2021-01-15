# W-kernel support size

## Introduction

To reserve sufficient space for the convolution by the w-kernel it is important to know the support in advance. This document describes the derivation of an approximate formula for the w-kernel support size

When the W-term is Nyquist sampled in the image domain, the support in the uv-domain fits exactly inside the grid. Strictly speaking the W-term can not be Nyquist sampled, because its support is infinite. However, the W-term can be considered Nyquist sampled when the phase change from pixel to pixel is smaller then $`\pi`$. This criterion cuts the tails of the convolution function at approximately the 1% level.

## Derivation

Let the phase of the W-term be given as function of image coordinate $`l`$,
```math
   \varphi(l) = 2\pi w(1-n(l)) = 2\pi w(1-\sqrt{1 - l^2}),
```
where $`w`$ is the w-coordinate in wavelengths.

The phase change from pixel to pixel is given by
```math
    \Delta\varphi = \frac{\mathrm{d}\varphi(l)}{\mathrm{d}l}\Delta l,
```
where $`\Delta l = s/N `$ and $`s`$ is the image size in radians and $`N`$ is the image size in number of pixels.

The derivative is given by
```math
    \frac{\mathrm{d}\varphi(l)}{\mathrm{d}l} = -2\pi w \frac{-l}{\sqrt{1 - l^2}}.
```
This can be simplified by using the approximation
```math
    \sqrt{1 - l^2} \approx 1
```
for small $`l`$.
And thus
```math
    \frac{\mathrm{d}\varphi(l)}{\mathrm{d}l} \approx 2\pi w l.
```

The W-term changes fastest at the edge of the image where $`l = s/2`$. At the edge 
```math
    \Delta \varphi = \pi w s^2/N
```

Equating $`\Delta \varphi`$ to $`\pi`$ leads to the final result $`N = w s^2`$.

## Introducing the shift

When imaging a facet in a tangent plane not for the center of the facet, the W-term will depend on the shift parameter as well.

Let the shift be given by $`\delta l`$. At the edge of the facet $`l' = s/2 + \delta l`$.

```math
    \Delta\varphi = 2\pi w l' \Delta l = 2\pi w (s/2 + \delta l) s/N
```

Equation this to $`\pi`$ leads to

```math
    N = w (s + 2\delta l)s
```
