import os
from typing import List, Union

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from fewshot_lib.config import EpisodeDescriptionConfig, DataConfig
from fewshot_lib.dataspec import BiLevelDatasetSpecification as BDS
from fewshot_lib.dataspec import DatasetSpecification as DS
from fewshot_lib.dataspec import Split
from fewshot_lib.sampler import EpisodeDescriptionSampler
from fewshot_lib.tfrecord import TFRecordDataset
from fewshot_lib.transform import get_transforms


class Reader(object):
    """Class reading data from one source and assembling examples.

    Specifically, it holds part of a tf.data pipeline (the source-specific part),
    that reads data from TFRecords and assembles examples from them.
    """

    def __init__(self,
                 dataset_spec: Union[BDS, DS],
                 split: Split,
                 shuffle: bool,
                 offset: int):
        """Initializes a Reader from a source.

        The source is identified by dataset_spec and split.

        Args:
          dataset_spec: DatasetSpecification, data_lib specification.
          split: A learning_spec.Split object identifying the source split.
        """
        self.dataset_spec = dataset_spec
        self.offset = offset
        self.shuffle = shuffle

        self.base_path = self.dataset_spec.path
        self.class_set = self.dataset_spec.get_classes(split)
        self.num_classes = len(self.class_set)

    def construct_class_datasets(self):
        """Constructs the list of class datasets.

        Returns:
          class_datasets: list of tf.data.Dataset, one for each class.
        """
        file_pattern = self.dataset_spec.file_pattern
        # We construct one data_lib object per class. Each data_lib outputs a stream
        # of `(example_string, dataset_id)` tuples.
        class_datasets = []
        for dataset_id in range(self.num_classes):
            class_id = self.class_set[dataset_id]  # noqa: E111
            if file_pattern.startswith('{}_{}'):
                # TODO(lamblinp): Add support for sharded files if needed.
                raise NotImplementedError('Sharded files are not supported yet. '  # noqa: E111
                                          'The code expects one data_lib per class.')
            elif file_pattern.startswith('{}'):
                data_path = os.path.join(self.base_path, file_pattern.format(class_id))  # noqa: E111
                index_path = os.path.join(self.base_path, '{}.index'.format(class_id))  # noqa: E111
            else:
                raise ValueError('Unsupported file_pattern in DatasetSpec: %s. '  # noqa: E111
                                 'Expected something starting with "{}" or "{}_{}".' %
                                 file_pattern)
            description = {"image": "byte", "label": "int"}

            dataset = TFRecordDataset(data_path=data_path,
                                      index_path=index_path,
                                      description=description,
                                      shuffle=self.shuffle)

            class_datasets.append(dataset)

        assert len(class_datasets) == self.num_classes
        return class_datasets

def cycle_(iterable):
    # Creating custom cycle since itertools.cycle attempts to save all outputs in order to
    # re-cycle through them, creating amazing memory leak
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def decode_feat_dic(feat_dic):
    # get BGR image from bytes
    image = cv2.imdecode(feat_dic["image"], -1)
    # from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    feat_dic["image"] = image
    return feat_dic


def worker_init_fn_(worker_id, seed):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the data_lib copy in this worker process
    random_gen = np.random.RandomState(seed + worker_id)
    dataset.random_gen = random_gen
    for source_dataset in dataset.dataset_list:
        source_dataset.random_gen = random_gen
        for class_dataset in source_dataset.class_datasets:
            class_dataset.random_gen = random_gen

def make_episode_pipeline(dataset_spec_list: List[Union[BDS, DS]],
                          split: Split,
                          episode_descr_config: EpisodeDescriptionConfig,
                          data_config: DataConfig,
                          ignore_hierarchy_probability: int = 0.0,
                          **kwargs):
    """Returns a pipeline emitting data from potentially multiples source as Episodes.

    Args:
      dataset_spec_list: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.

    Returns:
    """

    episodic_dataset_list = []
    for i in range(len(dataset_spec_list)):
        episode_reader = Reader(dataset_spec=dataset_spec_list[i],
                                split=Split["TRAIN"],
                                shuffle=data_config.shuffle,
                                offset=0)
        class_datasets = episode_reader.construct_class_datasets()
        sampler = EpisodeDescriptionSampler(
            dataset_spec=episode_reader.dataset_spec,
            split=Split["TRAIN"],
            episode_descr_config=episode_descr_config,
            use_bilevel_hierarchy=episode_descr_config.use_bilevel_ontology_list[i])
        transforms = get_transforms(data_config, split)
        _, max_support_size, max_query_size = sampler.compute_chunk_sizes()
        episodic_dataset_list.append(EpisodicDataset(class_datasets=class_datasets,
                                                     sampler=sampler,
                                                     max_support_size=max_support_size,
                                                     max_query_size=max_query_size,
                                                     transforms=transforms))

    return ZipDataset(episodic_dataset_list)


def make_batch_pipeline(dataset_spec_list: List[Union[BDS, DS]],
                        data_config: DataConfig,
                        split: Split,
                        **kwargs):
    """Returns a pipeline emitting data from potentially multiples source as batches.

    Args:
      dataset_spec_list: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
    Returns:
    """

    offset = 0
    dataset_list = []
    for dataset_spec in dataset_spec_list:
        batch_reader = Reader(dataset_spec=dataset_spec,
                                     split=Split["TRAIN"],
                                     shuffle=data_config.shuffle,
                                     offset=offset)

        class_datasets = batch_reader.construct_class_datasets()

        transforms = get_transforms(data_config=data_config, split=split)
        dataset = BatchDataset(class_datasets=class_datasets,
                               transforms=transforms)
        dataset_list.append(dataset)
        offset += len(class_datasets)
    dataset = ZipDataset(dataset_list)
    return dataset


class EpisodicDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 sampler: EpisodeDescriptionSampler,
                 transforms: torchvision.transforms,
                 max_support_size: int,
                 max_query_size: int):
        super(EpisodicDataset).__init__()
        self.class_datasets = class_datasets
        self.sampler = sampler
        self.transforms = transforms
        self.max_query_size = max_query_size
        self.max_support_size = max_support_size
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            episode_description = self.sampler.sample_episode_description(self.random_gen)
            support_images = []
            support_labels = []
            query_images = []
            query_labels = []
            episode_classes = list({class_ for class_, _, _ in episode_description})
            for class_id, nb_support, nb_query in episode_description:
                used_ids = []
                sup_added = 0
                query_added = 0
                while sup_added < nb_support:
                    sample_dic = self.get_next(class_id)
                    if sample_dic['id'] not in used_ids:
                        used_ids.append(sample_dic['id'])
                        support_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                        sup_added += 1
                while query_added < nb_query:
                    sample_dic = self.get_next(class_id)
                    if sample_dic['id'] not in used_ids:
                        used_ids.append(sample_dic['id'])
                        query_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                        query_added += 1
                # print(f"Class {class_id} contains duplicate: {contains_duplicates(used_ids)}")
                support_labels.extend([episode_classes.index(class_id)] * nb_support)
                query_labels.extend([episode_classes.index(class_id)] * nb_query)
            support_images = torch.cat(support_images, 0).cpu()
            query_images = torch.cat(query_images, 0).cpu()
            support_labels = torch.tensor(support_labels).cpu()
            query_labels = torch.tensor(query_labels).cpu()
            yield support_images, query_images, support_labels, query_labels

    def __len__(self):
        return sum(len(dataset) for dataset in self.class_datasets)

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])
        sample_dic = decode_feat_dic(sample_dic)
        return sample_dic

class BatchDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 transforms: torchvision.transforms):
        super(BatchDataset).__init__()
        self.class_datasets = class_datasets
        self.transforms = transforms

    def __iter__(self):
        while True:
            rand_class = self.random_gen.randint(len(self.class_datasets))
            sample_dic = self.get_next(rand_class)
            sample_dic = decode_feat_dic(sample_dic)
            transformed_image = self.transforms(sample_dic['image'])
            target = sample_dic['label'][0]
            yield transformed_image, target

    def __len__(self):
        return sum(len(dataset) for dataset in self.class_datasets)

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])
        return sample_dic


class ZipDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        dataset_list: List[EpisodicDataset]):
        self.dataset_list = dataset_list
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            rand_source = self.random_gen.randint(len(self.dataset_list))
            next_e = self.get_next(rand_source)
            yield next_e
    def get_next(self, source_id):
        try:
            dataset = next(self.dataset_list[source_id])
        except:
            self.dataset_list[source_id] = iter(self.dataset_list[source_id])
            dataset = next(self.dataset_list[source_id])
        return dataset
