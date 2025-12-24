# Free Testosterone Model
This model estimates the free testosterone from total testosterone, SHBG, and albumin in a blood sample. I built it using raw data from public datasets, compared it to the established [Vermeulen formula](https://doi.org/10.1210/jcem.84.10.6079), and validated it for clinical diagnostic usability against equilibrium dialysis and ultrafiltration (gold-standard) references.
### What does testosterone in the blood look like?

When we look at testosterone data, we often see a lot of noise. In our blood, testosterone exists in three different forms:
- Free testosterone is unbound and biologically active, meaning it is able to enter cells
- SHBG-bound testosterone is tightly bound, and thus unavailable to tissue cells
- Albumin-bound testosterone is weakly bound and only somewhat biologically active

Altogether, these forms sum up to **total testosterone**.

### Bioavailable vs. free testosterone
The bioavailable testosterone, as you may have noticed, is composed of both the free and albumin-bound forms. In clinical settings, we typically only focus on the free testosterone, which correlates best with symptom explanation. Albumin-bound testosterone *is* technically biologically active, but only once it becomes free (and can thus interact with tissue cells).

The big problem in clinical diagnostics is that free testosterone is **hard/expensive to directly measure**. Labs can use equilibrium dialysis (ED) or ultrafiltration, but both methods are inaccessible to low-resource clinics and hospitals.

### How do we measure and calculate free testosterone?

In order to find free testosterone (to calculate our total bioavailable testosterone), we first measure the **total testosterone**. This can be done using the common method of using an immunoassay, or through a more expensive (but accurate) process called “liquid chromatography–mass spectrometry”, or LC-MS/MS. SHBG and albumin are large proteins that appear in large concentrations in the blood, making it easy to measure in simple immunoassay. Once we have those three factors, we can solve for free testosterone using a mathematical model.

Currently, the Vermeulen model offers the [most robust](https://doi.org/10.1210/jc.2017-02360) method of estimating free testosterone, with the best correlation to measured values.

### What can testosterone measurements tell us?
Blood free testosterone values help us understand androgen signaling capacity, or the strength of the hormone’s signal. Free testosterone crosses cell membranes and binds to androgen receptors, driving androgen-dependent processes. So when free testosterone is **low**, we might see:

- Reduced muscle protein synthesis
- Reduced libido
- Lower red blood cell count
- Slower bone accrual

When it’s **high**, we might see excessive androgen signaling, resulting in:

- Increased sebum production, or acne
- Increased red blood cell production
- Increased drive and restlessness
- Impulsive behaviors under stress

### How can I use this model?


## Building the model
I started by deriving a quadratic equation for free testosterone (FT) from the mass action equation, which is: 

$TT = FT + \text{SHBG-bound} + \text{Albumin-bound}$

We can also write our binding equilibria with Langmuir/Michaelis-Menten kinetics built-in. This accounts for the selectivity of biochemical reactions that involve a single substrate:

$$\text{SHBG-bound}=\frac{[\text{SHBG}]_{\text{total}}\cdot K_{\text{SHBG}} \cdot FT}{1 + K_{\text{SHBG}} \cdot FT}$$

$\text{Albumin-bound}=K_{\text{ALB}} \cdot [\text{ALB}] \cdot FT$

Now if we substitute our variables in the mass balance, we get:

$TT=FT+\frac{[\text{SHBG}] \cdot K_{\text{SHBG}} \cdot FT}{1 + K_{\text{SHBG}} \cdot FT}+K_{\text{ALB}} \cdot [\text{ALB}] \cdot FT$

>Some notes here: the Vermeulen model simplifies the SHBG binding process by assuming a single binding constant, $K_{\text{SHBG}}$, even though SHBG has two binding sites. This simplification was addressed by the [Zakharov model](https://doi.org/10.1016/j.mce.2014.09.001) in 2015, but the Vermeulen still remains the clinical favorite.

Our model does *not* factor in competitors, like DHT or E2, so our resulting equation is quadratic (rather than cubic).
### Physical Bounds
Free testosterone must be positive, but also less than total testosterone. Thus, we can express FT constraints as:
$0<FT<TT$
The binding function is also increasing in free testosterone, meaning that the equation is **monotonic** (one, unique solution). This can be expressed through the differential proof:

$\frac{dFT}{dTT}=\frac{1}{(FT+\frac{b}{2a})+\sqrt{(FT+\frac{b}{2a})^2+FT^2}}>0$

### Association Constants & Units
I collected the following parameters from Vermeulen et al. (1999):
- $K_{\text{SHBG}} = 1.0×10^9$
- $K_{\text{ALB}} = 3.4×10^4$

**Units**
-   SHBG: nmol/L (direct)
-   Albumin: g/L → mol/L via MW ≈ 66,500 g/mol
-   Testosterone: nmol/L (or ng/dL ÷ 28.84)







