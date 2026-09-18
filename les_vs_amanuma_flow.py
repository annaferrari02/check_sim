"""
Confronto della FRAZIONE di flusso rispetto all'inlet (sopraceliaco = SC).
Grandezza: IR/SC (= LivelloC / LivelloA), cioe' quota di flusso che raggiunge
l'infrarenale. Normalizzando per l'inlet si rimuove la scala legata a
taglia/eta' (gittata cardiaca) che confondeva il confronto sui valori assoluti.

Regola d'oro: il rapporto va calcolato PER SOGGETTO e poi mediato.
    media(C)/media(A)  e' SBAGLIATO (nessuna SD valida, perde l'accoppiamento).

Dati:
  - Amanuma et al. 1992, Tab.1: valori individuali A e C (n=10) -> rapporti diretti.
  - Les et al. 2010, Tab.1: rapporto IR/SC gia' per-paziente = 0.343 +/- 0.0843 (n=36).
"""

import numpy as np
from scipy import stats

ALPHA = 0.05


# Amanuma 1992 - dati individuali (ml/min). Ordine: soggetti 1..10
A = np.array([2226, 2789, 3350, 4639, 3461, 3628, 8001, 2761, 4807, 5275])  # SC / Livello A
B = np.array([1179, 1312, 2363, 3767, 2947, 2519, 4493, 2593, 2931, 3673])  # Livello B
C = np.array([ 685,  612, 1550, 1934, 1069, 2735, 2009, 1772, 1173, 1781])  # IR / Livello C

ratio_CA = C / A                 # frazione IR/SC per soggetto
ratio_BA = B / A                 # frazione B/SC (informativa: nessun omologo in Les)

# Les 2010 - rapporto IR/SC gia' per-paziente
les_ratio_mean, les_ratio_sd, les_n = 0.343, 0.0843, 36


def welch(m1, s1, n1, m2, s2, n2, alpha=ALPHA):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    v1, v2 = s1**2 / n1, s2**2 / n2
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    diff = m1 - m2
    tc = stats.t.ppf(1 - alpha / 2, df)
    ci = (diff - tc * se, diff + tc * se)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = diff / sp
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    return dict(t=t, p=p, df=df, diff=diff, ci=ci, d=d, g=g)


def riga(nome, r):
    sig = "SI" if r["p"] < ALPHA else "no"
    print(f"  {nome}")
    print(f"    diff (sani-AAA) = {r['diff']:+.4f}   IC95% [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    print(f"    Welch: t={r['t']:.3f}  df={r['df']:.2f}  p={r['p']:.4f}  -> sig 0.05: {sig}")
    print(f"    effect size: Hedges g={r['g']:.3f}")


print("=" * 72)
print("FRAZIONE DI FLUSSO SULL'INLET  —  IR/SC  —  sani (Amanuma) vs AAA (Les)")
print("=" * 72)

print("\nRapporti IR/SC individuali (Amanuma):")
for i in range(len(C)):
    print(f"soggetto {i+1:2d}: {C[i]/A[i]:.3f}")
    if C[i]>A[i]:
        print(f"  soggetto {i+1:2d}: {C[i]/A[i]:.3f}  <-- soggetto anomalo (C>A)")


# --- confronto principale: tutti i 10 soggetti ---
m1, s1, n1 = ratio_CA.mean(), ratio_CA.std(ddof=1), len(ratio_CA)
print(f"\nAmanuma IR/SC (n={n1}): {m1:.3f} +/- {s1:.3f}")
print(f"Les     IR/SC (n={les_n}): {les_ratio_mean:.3f} +/- {les_ratio_sd:.3f}")

print("\n[Principale] tutti i 10 soggetti Amanuma:")
riga("IR/SC", welch(m1, s1, n1, les_ratio_mean, les_ratio_sd, les_n))

# escludo soggetto 6 perche ha C > B come è possibile? 
ratio_CA_no6 = np.delete(ratio_CA, 5)
m1b, s1b, n1b = ratio_CA_no6.mean(), ratio_CA_no6.std(ddof=1), len(ratio_CA_no6)
print(f"\n[Sensitivity] escluso soggetto 6  (Amanuma: {m1b:.3f} +/- {s1b:.3f}, n={n1b}):")
riga("IR/SC", welch(m1b, s1b, n1b, les_ratio_mean, les_ratio_sd, les_n))


print(f"\n[Informativa] frazione B/SC Amanuma (no omologo in Les): "
      f"{ratio_BA.mean():.3f} +/- {ratio_BA.std(ddof=1):.3f}")

## proviamo a vedere allora il livello B cosa ci dice (quindi, il rapporto B/A) dato che potenzialmente portebbe essere il criterio per 
##suddividere il flusso tra celiactrunk+mesenterica sup vs renali dx e sx 


f_visc  = (A - B) / A   # tripode + SMA  (prelievo tra A e B)
f_renal = (B - C) / A   # renali dx+sx   (prelievo tra B e C)
f_infra =  C / A        # residuo infrarenale (oltre C)

print(f"{'sogg':>4} {'visc(CA+SMA)':>13} {'renali':>8} {'infra':>7} {'somma':>7}")
for i in range(len(A)):
    tot = f_visc[i] + f_renal[i] + f_infra[i]
    flag = ""
    if B[i] > A[i]:      flag = " <-- B>A impossibile"
    elif C[i] > B[i]:    flag = " <-- renale<0 impossibile (C>B)"
    print(f"{i+1:>4} {f_visc[i]:>13.3f} {f_renal[i]:>8.3f} {f_infra[i]:>7.3f} {tot:>7.3f}{flag}")



#confronto con Les 2010
lit = dict(SC=3.51, IR=1.31, CT=0.33, SMA=0.223, LR=0.223, RR=0.223)
lit["UBVF"] = lit["SC"] - lit["IR"]
ref_visc_ubvf  = lit["CT"] + lit["SMA"]                 # 0.553
ref_renal_ubvf = lit["LR"] + lit["RR"]                  # 0.446
ubvf_over_sc   = lit["UBVF"] / lit["SC"]                # 0.627
ref_visc_sc    = ref_visc_ubvf  * ubvf_over_sc          # 0.347
ref_renal_sc   = ref_renal_ubvf * ubvf_over_sc          # 0.280
ref_infra_sc   = lit["IR"] / lit["SC"]                  # 0.373
 
 
def confronta_con_riferimento(nome, x, ref, alpha=0.05):
    """t-test a un campione: media di x (Amanuma) vs costante 'ref' (letteratura)."""
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    t, p = stats.ttest_1samp(x, ref)
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    se = sd / np.sqrt(n)
    ci = (m - tc * se, m + tc * se)
    sig = "SI" if p < alpha else "no"
    print(f"  {nome:22s} Amanuma {m:.3f} +/- {sd:.3f} (n={n})  "
          f"IC95%[{ci[0]:.3f}, {ci[1]:.3f}]  vs rif {ref:.3f}  "
          f"-> t={t:+.2f} p={p:.3f}  sig:{sig}")
    return dict(mean=m, sd=sd, n=n, ref=ref, t=t, p=p, ci=ci)
 
valid = (C <= B) & (B <= A)
# --- frazioni Amanuma (solo soggetti validi) ---
Av, Bv, Cv = A[valid], B[valid], C[valid]
visc_ubvf  = (Av - Bv) / (Av - Cv)
renal_ubvf = (Bv - Cv) / (Av - Cv)
visc_sc    = (Av - Bv) / Av
renal_sc   = (Bv - Cv) / Av
infra_sc   = Cv / Av
 
print(f"Soggetti esclusi (split impossibile): {list(np.where(~valid)[0] + 1)}")
 
print("\n[Base UBVF]  spartizione del flusso di ramo (viscerale vs renale):")
confronta_con_riferimento("viscerale (CA+SMA)", visc_ubvf,  ref_visc_ubvf)
confronta_con_riferimento("renale (dx+sx)",     renal_ubvf, ref_renal_ubvf)
 
print("\n[Base SC]  frazioni sull'inlet:")
confronta_con_riferimento("viscerale (CA+SMA)", visc_sc,  ref_visc_sc)
confronta_con_riferimento("renale (dx+sx)",     renal_sc, ref_renal_sc)
confronta_con_riferimento("infrarenale residuo", infra_sc, ref_infra_sc)
 
print("\nNOTA bene :")
print(" - siamo senza SD: il test conta solo la variabilita' campionaria di Amanuma.")
print("   'non significativo' = Amanuma")
print("   compatibile con la BC assunta, non prova di identita'.")
