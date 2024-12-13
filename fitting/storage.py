

def createDataFromPosterior(bin_counts, num):
    dist = torch.distributions.multinomial.Multinomial(
        total_count=num, probs=bin_counts
    )
    return dist.sample()


if __name__ == "__main__":
    bkg_data = torch.load(
        "gaussian_window_results/signal_312_1500_400/inject_r_0p0/bkg_estimation_result.pth"
    )
    hist = bkg_data["input_data"]
    obs, pred = getPrediction(bkg_data, model_class=fitting.models.NonStatParametric2D)
    print(obs.Y.sum())
    print(torch.max(obs.Y))
    print(torch.max(pred.mean))
    print(pred.mean.sum())
    h = createDataFromPosterior(torch.clamp(pred.mean, min=0), int(pred.mean.sum()))
    print(torch.max(h))
    print(h.sum())

    good_bin_mask = h > 10
    global_chi2_bins = chi2Bins(pred.mean, h, torch.sqrt(h), good_bin_mask)
    print(global_chi2_bins)
