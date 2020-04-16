#!/usr/bin/env python3
"""Main function for training and testing."""

import opts, train, dataloader, utils, evaluation, plot
import init, checkpoints, criterion
import numpy as np
import os
import sys
import torch
from torch.nn import functional as F


def main():
    """Main function for training and testing."""
    # Parse command line arguments and cache
    opt = opts.Opts().args
    utils.savecmd(opt.resume, sys.argv)

    utils.print_color_msg("==> Setting up data loader")
    train_loader, val_loader, test_loader = dataloader.create(opt)

    # Load checkpoint if specified, None otherwise
    utils.print_color_msg("==> Checking checkpoints")
    checkpoint = checkpoints.load(opt)

    utils.print_color_msg("==> Setting up model and criterion")
    model, optim_state = init.setup(opt, checkpoint)
    loss_fn = criterion.setup(opt, checkpoint)

    utils.print_color_msg("==> Loading trainer")
    trainer = train.create_trainer(model, loss_fn, opt, optim_state)

    best_loss = float('Inf')
    val_loss = float('Inf')
    start_epoch = max([1, opt.epochNum])
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print("".ljust(4) + "Previous best loss: "
              + utils.color_msg('%.5f' %best_loss))

    if opt.valOnly:
        assert start_epoch > 1, "There must be at least one epoch"
        utils.print_color_msg("==> Validation:")
        print("".ljust(4) + "=> Epoch %i" %(start_epoch-1))
        trainer.val(val_loader, start_epoch-1)
        sys.exit()

    if opt.testOnly:
        assert start_epoch > 1, "There must be at least one epoch"
        utils.print_color_msg("==> Testing:")
        print("".ljust(4) + "=> Epoch %i" %(start_epoch-1))
        _, prediction, reference, post , seq_length = trainer.test(test_loader, start_epoch-1)

        prediction = F.sigmoid(torch.Tensor(prediction)).numpy()
        nce = evaluation.nce(reference, prediction)
        precision, recall, area, threshold  = evaluation.pr(reference, prediction)
        precision_bl, recall_bl, area_bl, _ = evaluation.pr(reference, post)
        f1, f1_precision, f1_recall, f1_threshold = evaluation.f1(precision,recall,threshold)
        tpr, fpr, roc_area = evaluation.roc(reference, prediction)

        # Calculate stats for sequences binned by the posterior
        limits = np.linspace(0,1,11).tolist()
        utils.print_color_msg('\n\nEffect of Input Posterior on Performance')
        for i in range(len(limits)-1):
            ref, pred, p = evaluation.bin_results(reference, prediction, post, measure=post, \
                                                  lower_limit=limits[i], upper_limit=limits[i+1])
            if ref.size:
                nce_post = evaluation.nce(ref, pred)
                nce_post_bl = evaluation.nce(ref, p)
                precision_post, recall_post, area_post, threshold_post = evaluation.pr(ref, pred)
                precision_post_bl, recall_post_bl, area_post_bl, threshold_post_bl = evaluation.pr(ref, p)
                f1_post, _, _, _ = evaluation.f1(precision_post, recall_post, threshold_post)
                f1_post_bl, _, _, _ = evaluation.f1(precision_post_bl, recall_post_bl, threshold_post_bl)
                _, _, roc_area_post = evaluation.roc(ref, pred)
                print('%.1f. - %.1f. %d    Results (model/bl)     NCE: %.4f. , %.4f.    AUC(PR): %.4f. , %.4f.    F-1:  %.4f. , %.4f.    AUC(ROC): %.4f.'\
                      %(limits[i], limits[i+1], int(ref.size), nce_post, nce_post_bl, area_post, area_post_bl, f1_post, f1_post_bl, roc_area_post))
            else:
                print('%.1f. - %.1f. Empty' %(limits[i],limits[i+1]))

        # Caluclate stats for sequences binned by sequence length
        limits = [0,2,3,6,10,20,40]
        utils.print_color_msg('\n\nEffect of Sequence Length on Performance')
        for i in range(len(limits)-1):
            ref, pred, p = evaluation.bin_results(reference, prediction, post, measure=seq_length, \
                                                  lower_limit=limits[i], upper_limit=limits[i+1])
            if ref.size:
                nce_len = evaluation.nce(ref, pred)
                nce_len_bl = evaluation.nce(ref, p)
                precision_len, recall_len, area_len, threshold_len = evaluation.pr(ref, pred)
                precision_len_bl, recall_len_bl, area_len_bl, threshold_len_bl = evaluation.pr(ref, p)
                f1_len, _, _, _ = evaluation.f1(precision_len, recall_len, threshold_len)
                f1_len_bl, _, _, _ = evaluation.f1(precision_len_bl, recall_len_bl, threshold_len_bl)
                _, _, roc_area_len = evaluation.roc(ref, pred)
                print(f'%d - %d  %d   Results (model/bl)    NCE: %.4f. , %.4f.    AUC: %.4f. , %.4f.    F-1:  %.4f. , %.4f.    AUC(ROC): %.4f.'\
                      %(limits[i], limits[i+1], int(ref.size), nce_len, nce_len_bl, area_len, area_len_bl, f1_len, f1_len_bl, roc_area_len))
            else:
                print('%d - %d Empty' %(limits[i],limits[i+1]))

        # Calulate calibration stats
        limits = np.linspace(0,1,11).tolist()
        print('\n\nCalibration Stats')
        ece = 0
        for i in range(len(limits)-1):
            ref, pred, p = evaluation.bin_results(reference, prediction, post, measure=prediction, \
                                                  lower_limit=limits[i], upper_limit=limits[i+1])
            if ref.size:
                accuracy_bin = np.mean(ref)
                confidence_bin = np.mean(pred)
                posterior_bin = np.mean(p)
                ece += abs(accuracy_bin - confidence_bin)*len(ref)/len(reference)
                print(f'%.1f. - %.1f. %d    Reference: %.4f. ,    Prediction: %.4f. ,    Posterior: %.4f.'
                      %(limits[i], limits[i+1], int(ref.size), accuracy_bin, confidence_bin, posterior_bin))
            else:
                print('%.1f. - %.1f. Empty' %(limits[i], limits[i+1]))

        # Print Test Stats
        print('\n\nTest Stats')
        print(
            "".ljust(7) + "\nNCE: %.4f. \nAUC(PR): %.4f. \nF-1: %.4f. p: %.4f. r: %.4f. t: %.4f. \nAUC(ROC): %.4f. \nECE: %.4f. " \
            %(nce, area, f1, f1_precision, f1_recall, f1_threshold, roc_area, nce))

        trainer.logger['test'].write('NCE: %f\nAUC(PR): %f\n' %(nce, area))
        evaluation.plot_pr([precision, precision_bl], [recall, recall_bl],
                           [area, area_bl], ['BiLatticeRNN', 'posterior'],
                           opt.resume)
        np.savez(os.path.join(opt.resume, 'result.npz'),
                 prediction=prediction, reference=reference, posteriors=post)
        sys.exit()

    utils.print_color_msg("==> Training:")
    for epoch in range(start_epoch, opt.nEpochs+1):
        print("".ljust(4) + "=> Epoch %i" %epoch)
        best_model = False
        _ = trainer.train(train_loader, epoch, val_loss)

        if not opt.debug:
            val_loss = trainer.val(val_loader, epoch)
            if val_loss < best_loss:
                best_model = True
                print("".ljust(4) + "** Best model: "
                      + utils.color_msg('%.4f' %val_loss))
                best_loss = val_loss
            checkpoints.save(epoch, trainer.model, loss_fn,
                             trainer.optim_state, best_model, val_loss, opt)

    if not opt.debug:
        utils.print_color_msg("==> Testing:")
        _, prediction, reference, _ , _ = trainer.test(test_loader, opt.nEpochs)
        prediction = F.sigmoid(torch.Tensor(prediction)).numpy()
        nce = evaluation.nce(reference, prediction)
        precision, recall, area, _ = evaluation.pr(reference, prediction)
        utils.print_color_msg(
            "".ljust(7) + "NCE: %.4f. AUC(PR): %.4f" %(nce, area))
        trainer.logger['test'].write('NCE: %f\nAUC(PR): %f\n' %(nce, area))
        evaluation.plot_pr([precision], [recall], [area], ['BiLatticeRNN'], opt.resume)

        # Flush write out and reset pointer
        for open_file in trainer.logger.values():
            open_file.flush()
            open_file.seek(0)
        plot.plot(opt.resume, opt.onebest)

if __name__ == '__main__':
    main()
