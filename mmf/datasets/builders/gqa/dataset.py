import torch

from mmf.datasets.builders.vqa2.dataset import VQA2Dataset


class GQADataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="gqa",
            *args,
            **kwargs
        )

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)

            predictions.append(
                {
                    "questionId": question_id.item(),
                    "prediction": answer,
                }
            )

        return predictions
