Setup

A 1D Gaussian with spatial variance σ²:
$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)$$

Its Fourier transform (convention $F(\omega) = \int f(x), e^{-i\omega x}, dx$):
$$F(\omega) = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right) e^{-i\omega x}, dx$$

Complete the square in the exponent

$$-\frac{x^2}{2\sigma^2} - i\omega x = -\frac{1}{2\sigma^2}\left(x^2 + 2i\sigma^2\omega x\right) = -\frac{(x + i\sigma^2\omega)^2}{2\sigma^2} - \frac{\sigma^2\omega^2}{2}$$

Integrate. The first term integrates to the ordinary Gaussian integral $\sqrt{2\pi\sigma^2}$ (shifting the contour off the real axis by a constant imaginary amount doesn't change the value, since the integrand is entire and decays — Cauchy's theorem):

$$F(\omega) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \sqrt{2\pi\sigma^2}, \exp\left(-\frac{\sigma^2\omega^2}{2}\right) = \exp\left(-\frac{\sigma^2 \omega^2}{2}\right)$$

Read off the frequency-domain width. Write this in the same Gaussian form, $\exp(-\omega^2 / 2\sigma_\omega^2)$:

$$\sigma_\omega^2 = \frac{1}{\sigma^2} \quad\Longleftrightarrow\quad \sigma_\omega = \frac{1}{\sigma}$$

So spatial standard deviation σ and frequency-domain standard deviation σ_ω are exact reciprocals — a narrow spatial peak (small σ) produces a wide frequency-domain spread (large σ_ω = 1/σ), and vice versa. This is the Gaussian case of the general time/frequency uncertainty relation σ_x · σ_ω ≥ 1 (Gaussians hit equality — they're the minimum-uncertainty case).

Matching the codebase's convention. T2FFT (T2FFT.py:12-13) uses cycles, not angular frequency: exp(-i2π ξx). Redoing the same completed-square derivation with that convention picks up an extra 2π:

$$F(\xi) = \exp\left(-2\pi^2\sigma^2\xi^2\right) ;\Rightarrow; \sigma_\xi = \frac{1}{2\pi\sigma}$$
                                                                                              Same reciprocal scaling, just a 2π ention — the 1/σ relationship itself is convention-independent.                                                                    
Connecting to the 2D isotropic belief and r. For an isotropic belief on (x,y) with covariance σ²I, the Fourier transform factors ch 1D Gaussians, giving an isotropic Gaussian in (u,v) with the same 1/σ-scaled spread in every direction. Since the polar grid's r = √(u²+v²), the characteristic radiusnificant frequency content scalesas r_char ∝ 1/σ. Shrink σ (sharpen the belief) and r_char grows without bound — pushing more of the distribution's energy toward lawhere resample_c2p's bilinear/spline interpolation is weakest (established earlier). That's the precise mechanism behind "bias grows with sharpness."