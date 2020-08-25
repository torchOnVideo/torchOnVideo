from __future__ import print_function

from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, test, train
from config import TrainConfig as C
from ..loader.MSVD import MSVD
from ..loader.MSRVTT import MSRVTT
from ..models.decoder import Decoder
from ..models.caption_generator import CaptionGenerator



class TrainModel(SALSTM):
    def __init__(self, dataset, lr, ep):
        super(TrainModel, self).__init__()
        self.build_loaders(dataset, lr, ep)


    def build_loaders(self, dataset,lr,ep):
        self.C.corpus =dataset
        self.C.lr = lr
        self.C.ep = ep
        if self.C.corpus == "MSVD":
            self.corpus = MSVD(C)
        elif self.C.corpus == "MSR-VTT":
            self.corpus = MSRVTT(C)
        print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
            self.corpus.vocab.n_vocabs, self.corpus.vocab.n_vocabs_untrimmed, self.corpus.vocab.n_words,
            self.corpus.vocab.n_words_untrimmed, self.C.loader.min_count))
        return self.corpus.train_data_loader, self.corpus.val_data_loader, self.corpus.test_data_loader, self.corpus.vocab


    def log_train(summary_writer, e, loss, lr, scores=None):
        summary_writer.add_scalar(self.C.tx_train_loss, loss['total'], e)
        summary_writer.add_scalar(self.C.tx_train_cross_entropy_loss, loss['cross_entropy'], e)
        summary_writer.add_scalar(self.C.tx_train_entropy_loss, loss['entropy'], e)
        summary_writer.add_scalar(self.C.tx_lr, lr, e)
        print("loss: {} (CE {} + E {})".format(loss['total'], loss['cross_entropy'], loss['entropy']))
    
        if scores is not None:
            for metric in self.C.metrics:
                summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
            print("scores: {}".format(scores))
    
    
    def log_val(summary_writer, e, loss, scores):
        summary_writer.add_scalar(self.C.tx_val_loss, loss['total'], e)
        summary_writer.add_scalar(self.C.tx_val_cross_entropy_loss, loss['cross_entropy'], e)
        summary_writer.add_scalar(self.C.tx_val_entropy_loss, loss['entropy'], e)
        for metric in C.metrics:
            summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
        print("loss: {} (CE {} + E {})".format(loss['total'], loss['cross_entropy'], loss['entropy']))
        print("scores: {}".format(scores))
    
    
    def log_test(summary_writer, e, scores):
        for metric in self.C.metrics:
            summary_writer.add_scalar("TEST SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))


    def build_model(self,vocab):
        self.decoder = Decoder(
            rnn_type=self.decoder.rnn_type,
            num_layers=self.C.decoder.rnn_num_layers,
            num_directions=self.C.decoder.rnn_num_directions,
            feat_size=self.C.feat.size,
            feat_len=self.C.loader.frame_sample_len,
            embedding_size=self.C.vocab.embedding_size,
            hidden_size=self.C.decoder.rnn_hidden_size,
            attn_size=self.C.decoder.rnn_attn_size,
            output_size=self.vocab.n_vocabs,
            rnn_dropout=self.C.decoder.rnn_dropout)
        if self.C.pretrained_decoder_fpath is not None:
            self.decoder.load_state_dict(torch.load(self.C.pretrained_decoder_fpath)['decoder'])
            print("Pretrained decoder is loaded from {}".format(self.C.pretrained_decoder_fpath))
    
        self.model = CaptionGenerator(self.decoder, self.C.loader.max_caption_len, vocab)
        self.model.cuda()
        return self.model


    def __call__(self, *args, **kwargs):
        print("MODEL ID: {}".format(self.C.model_id))

        summary_writer = SummaryWriter(self.C.log_dpath)

        train_iter, val_iter, test_iter, vocab = self.build_loaders(dataset,lr,ep)

        model = self.build_model(vocab)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.C.lr, weight_decay=self.C.weight_decay, amsgrad=True)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.C.lr_decay_gamma,
                                         patience=self.C.lr_decay_patience, verbose=True)

        best_val_CIDEr = 0.
        best_epoch = None
        best_ckpt_fpath = None
        for e in range(1, C.epochs + 1):
            ckpt_fpath = C.ckpt_fpath_tpl.format(e)

            """ Train """
            print("\n")
            train_loss = train(e, self.model, optimizer, train_iter, vocab, self.C.decoder.rnn_teacher_forcing_ratio,
                               self.C.reg_lambda, self.C.gradient_clip)
            log_train(summary_writer, e, train_loss, get_lr(optimizer))

            """ Validation """
            val_loss = test(self.model, val_iter, vocab, self.C.reg_lambda)
            val_scores = evaluate(val_iter, self.model, vocab, beam_width=5, beam_alpha=0.)
            #log_val(summary_writer, e, val_loss, val_scores)

            if e >= self.C.save_from and e % C.save_every == 0:
                print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
                save_checkpoint(e, self.model, ckpt_fpath, self.C)

            if e >= self.C.lr_decay_start_from:
                lr_scheduler.step(val_loss['total'])
            if val_scores['CIDEr'] > best_val_CIDEr:
                best_epoch = e
                best_val_CIDEr = val_scores['CIDEr']
                best_ckpt_fpath = ckpt_fpath

        """ Test with Best Model """
        print("\n\n\n[BEST]")
        best_model = load_checkpoint(self.model, best_ckpt_fpath)
        best_scores = evaluate(test_iter, best_model, vocab, beam_width=5, beam_alpha=0.)
        print("scores: {}".format(best_scores))
        for metric in self.C.metrics:
            summary_writer.add_scalar("BEST SCORE/{}".format(metric), best_scores[metric], best_epoch)
        save_checkpoint(e, best_model, self.C.ckpt_fpath_tpl.format("best"), C)










if __name__ == "__main__":
    main()
