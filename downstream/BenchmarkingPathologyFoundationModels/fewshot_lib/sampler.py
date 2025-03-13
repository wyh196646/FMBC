from typing import Union, List, Tuple

import numpy as np
from absl import logging
from numpy.random import RandomState

from fewshot_lib import dataspec as dataset_spec_lib
from fewshot_lib.config import Split, EpisodeDescriptionConfig
from fewshot_lib.dataspec import BiLevelDatasetSpecification as BDS
from fewshot_lib.dataspec import DatasetSpecification as DS


def sample_num_ways_uniformly(random_gen: RandomState,
                              num_classes: int,
                              min_ways: int,
                              max_ways: int):
    """Samples a number of ways for an episode uniformly and at random.

    The support of the distribution is [min_ways, num_classes], or
    [min_ways, max_ways] if num_classes > max_ways.

    Args:
      num_classes: int, number of classes.
      min_ways: int, minimum number of ways.
      max_ways: int, maximum number of ways. Only used if num_classes > max_ways.

    Returns:
      num_ways: int, number of ways for the episode.
    """
    max_ways = min(max_ways, num_classes)
    return random_gen.randint(low=min_ways, high=max_ways + 1)


def sample_class_ids_uniformly(random_gen: RandomState,
                               num_ways: int,
                               rel_classes: List[int]):
    """Samples the (relative) class IDs for the episode.

    Args:
      num_ways: int, number of ways for the episode.
      rel_classes: list of int, available class IDs to sample from.

    Returns:
      class_ids: np.array, class IDs for the episode, with values in rel_classes.
    """
    return random_gen.choice(rel_classes, min(num_ways, len(rel_classes)), replace=False)


def compute_num_query(images_per_class: np.ndarray,
                      max_num_query: int,
                      num_support: Union[int, Tuple[int, int]]):
    """Computes the number of query examples per class in the episode.

    Query sets are balanced, i.e., contain the same number of examples for each
    class in the episode.

    The number of query examples satisfies the following conditions:
    - it is no greater than `max_num_query`
    - if support size is unspecified, it is at most half the size of the
      smallest class in the episode
    - if support size is specified, it is at most the size of the smallest class
      in the episode minus the max support size.

    Args:
      images_per_class: np.array, number of images for each class.
      max_num_query: int, number of images for each class.
      num_support: int or tuple(int, int), number (or range) of support
        images per class.

    Returns:
      num_query: int, number of query examples per class in the episode.
    """
    if num_support is None:
        if images_per_class.min() < 2:
            raise ValueError('Expected at least 2 images per class.')
        return np.minimum(max_num_query, (images_per_class // 2).min())
    elif isinstance(num_support, int):
        max_support = num_support
    else:
        _, max_support = num_support
    if (images_per_class - max_support).min() < 1:
        raise ValueError(
            'Expected at least {} images per class'.format(max_support + 1))
    return np.minimum(max_num_query, images_per_class.min() - max_support)


def sample_support_set_size(random_gen: RandomState,
                            num_remaining_per_class: np.ndarray,
                            max_support_size_contrib_per_class: int,
                            max_support_set_size: int):
    """Samples the size of the support set in the episode.

    That number is such that:

    * The contribution of each class to the number is no greater than
      `max_support_size_contrib_per_class`.
    * It is no greater than `max_support_set_size`.
    * The support set size is greater than or equal to the number of ways.

    Args:
      num_remaining_per_class: np.array, number of images available for each class
        after taking into account the number of query images.
      max_support_size_contrib_per_class: int, maximum contribution for any given
        class to the support set size. Note that this is not a limit on the number
        of examples of that class in the support set; this is a limit on its
        contribution to computing the support set _size_.
      max_support_set_size: int, maximum size of the support set.

    Returns:
      support_set_size: int, size of the support set in the episode.
    """
    if max_support_set_size < len(num_remaining_per_class):
        raise ValueError('max_support_set_size is too small to have at least one '
                         'support example per class.')
    beta = random_gen.uniform()
    support_size_contributions = np.minimum(max_support_size_contrib_per_class,
                                            num_remaining_per_class)
    return np.minimum(
        # Taking the floor and adding one is equivalent to sampling beta uniformly
        # in the (0, 1] interval and taking the ceiling of its product with
        # `support_size_contributions`. This ensures that the support set size is
        # at least as big as the number of ways.
        np.floor(beta * support_size_contributions + 1).sum(),
        max_support_set_size)


def sample_num_support_per_class(random_gen: RandomState,
                                 images_per_class: np.ndarray,
                                 num_remaining_per_class: np.ndarray,
                                 support_set_size: int,
                                 min_log_weight: float,
                                 max_log_weight: float):
    """Samples the number of support examples per class.

    At a high level, we wish the composition to loosely match class frequencies.
    Sampling is done such that:

    * The number of support examples per class is no greater than
      `support_set_size`.
    * The number of support examples per class is no greater than the number of
      remaining examples per class after the query set has been taken into
      account.

    Args:
      images_per_class: np.array, number of images for each class.
      num_remaining_per_class: np.array, number of images available for each class
        after taking into account the number of query images.
      support_set_size: int, size of the support set in the episode.
      min_log_weight: float, minimum log-weight to give to any particular class.
      max_log_weight: float, maximum log-weight to give to any particular class.

    Returns:
      num_support_per_class: np.array, number of support examples for each class.
    """
    if support_set_size < len(num_remaining_per_class):  # noqa: E111
        raise ValueError('Requesting smaller support set than the number of ways.')
    if np.min(num_remaining_per_class) < 1:  # noqa: E111
        raise ValueError('Some classes have no remaining examples.')

    # Remaining number of support examples to sample after we guarantee one
    # support example per class.
    remaining_support_set_size = support_set_size - len(num_remaining_per_class)  # noqa: E111

    unnormalized_proportions = images_per_class * np.exp(  # noqa: E111
        random_gen.uniform(min_log_weight, max_log_weight, size=images_per_class.shape))
    support_set_proportions = (  # noqa: E111
        unnormalized_proportions / unnormalized_proportions.sum())

    # This guarantees that there is at least one support example per class.
    num_desired_per_class = np.floor(  # noqa: E111
        support_set_proportions * remaining_support_set_size).astype('int32') + 1

    return np.minimum(num_desired_per_class, num_remaining_per_class)  # noqa: E111


class EpisodeDescriptionSampler(object):
    """Generates descriptions of Episode composition.  # noqa: E111

    In particular, for each Episode, it will generate the class IDs (relative to
    the selected split of the data_lib) to include, as well as the number of
    support and query examples for each class ID.
    """

    def __init__(self,  # noqa: E111
                 dataset_spec: Union[BDS, DS],
                 split: Split,
                 episode_descr_config: EpisodeDescriptionConfig,
                 use_bilevel_hierarchy: bool = False,
                 use_all_classes: bool = False):

        self.dataset_spec = dataset_spec
        self.split = split
        self.use_bilevel_hierarchy = use_bilevel_hierarchy
        self.use_all_classes = use_all_classes
        self.num_ways = episode_descr_config.num_ways
        self.num_support = episode_descr_config.num_support
        self.num_query = episode_descr_config.num_query
        self.min_ways = episode_descr_config.min_ways
        self.max_ways_upper_bound = episode_descr_config.max_ways_upper_bound
        self.max_num_query = episode_descr_config.max_num_query
        self.max_support_set_size = episode_descr_config.max_support_set_size
        self.max_support_size_contrib_per_class = episode_descr_config.max_support_size_contrib_per_class
        self.min_log_weight = episode_descr_config.min_log_weight
        self.max_log_weight = episode_descr_config.max_log_weight
        self.min_examples_in_class = episode_descr_config.min_examples_in_class

        self.class_set = dataset_spec.get_classes(self.split)
        self.num_classes = len(self.class_set)
        # Filter out classes with too few examples
        self._filtered_class_set = []
        # Store (class_id, n_examples) of skipped classes for logging.
        skipped_classes = []
        for class_id in self.class_set:
            n_examples = dataset_spec.get_total_images_per_class(class_id)  # noqa: E111
            if n_examples < self.min_examples_in_class:  # noqa: E111
                skipped_classes.append((class_id, n_examples))
            else:  # noqa: E111
                self._filtered_class_set.append(class_id)
        self.num_filtered_classes = len(self._filtered_class_set)

        if skipped_classes:
            logging.info(  # noqa: E111
                'Skipping the following classes, which do not have at least '
                '%d examples', self.min_examples_in_class)
        for class_id, n_examples in skipped_classes:
            logging.info('%s (ID=%d, %d examples)',  # noqa: E111
                         dataset_spec.class_names[class_id], class_id, n_examples)

        if self.min_ways and self.num_filtered_classes < self.min_ways:
            raise ValueError(  # noqa: E111
                '"min_ways" is set to {}, but split {} of data_lib {} only has {} '
                'classes with at least {} examples ({} total), so it is not possible '
                'to create an episode for it. This may have resulted from applying a '
                'restriction on this split of this data_lib by specifying '
                'benchmark.restrict_classes or benchmark.min_examples_in_class.'
                .format(self.min_ways, split, dataset_spec.name,
                        self.num_filtered_classes, self.min_examples_in_class,
                        self.num_classes))

        if self.use_all_classes:
            if self.num_classes != self.num_filtered_classes:  # noqa: E111
                raise ValueError('"use_all_classes" is not compatible with a value of '
                                 '"min_examples_in_class" ({}) that results in some '
                                 'classes being excluded.'.format(
                                   self.min_examples_in_class))
            self.num_ways = self.num_classes  # noqa: E111

        if episode_descr_config.ignore_bilevel_ontology:
            print("========================")
            print("Ignoring bilevel hierarchy !")
            print("========================")
            self.use_bilevel_hierarchy = False  # noqa: E111

        if self.use_bilevel_hierarchy:
            print("========================")
            print("Using bilevel hierarchy !")
            print("========================")
            # if self.num_ways is not None:  # noqa: E111
            #     raise ValueError('"use_bilevel_hierarchy" is incompatible with '
            #                      '"num_ways".')
            if self.min_examples_in_class > 0:  # noqa: E111
                raise ValueError('"use_bilevel_hierarchy" is incompatible with '
                                 '"min_examples_in_class".')

            if not isinstance(dataset_spec,  # noqa: E111
                              dataset_spec_lib.BiLevelDatasetSpecification):
                raise ValueError('Only applicable to datasets with a bi-level '
                                 'data_lib specification.')
            # The id's of the superclasses of the split (a contiguous range of ints).
            all_superclasses = dataset_spec.get_superclasses(self.split)  # noqa: E111
            self.superclass_set = []  # noqa: E111
            for i in all_superclasses:  # noqa: E111
                if self.dataset_spec.classes_per_superclass[i] < self.min_ways:
                    raise ValueError(  # noqa: E111
                        'Superclass: %d has num_classes=%d < min_ways=%d.' %
                        (i, self.dataset_spec.classes_per_superclass[i], self.min_ways))
                self.superclass_set.append(i)

    def sample_class_ids(self, random_gen: RandomState,):
        """Returns the (relative) class IDs for an episode.

        If self.use_dag_hierarchy, it samples them according to a procedure
        informed by the data_lib's ontology, otherwise randomly.
        If self.min_examples_in_class > 0, classes with too few examples will not
        be selected.
        """
        prob = [1.0, 0.0]

        if self.use_bilevel_hierarchy and random_gen.choice([True, False], p=prob):
            # First sample a coarse category uniformly. Then randomly sample the way
            # uniformly, but taking care not to sample more than the number of classes
            # of the chosen supercategory.
            episode_superclass = random_gen.choice(self.superclass_set, 1)[0]  # noqa: E111
            num_superclass_classes = self.dataset_spec.classes_per_superclass[  # noqa: E111
                episode_superclass]
            if self.num_ways:
                if self.num_ways > num_superclass_classes:
                    raise ValueError('{} classes required but superclass {} only contains {} classes'.format(
                                        self.num_ways,
                                        episode_superclass,
                                        num_superclass_classes))
                num_ways = self.num_ways
            else:
                num_ways = sample_num_ways_uniformly(  # noqa: E111
                    random_gen,
                    num_superclass_classes,
                    min_ways=self.min_ways,
                    max_ways=self.max_ways_upper_bound)

            # e.g. if these are [3, 1] then the 4'th and the 2'nd of the subclasses
            # that belong to the chosen superclass will be used. If the class id's
            # that belong to this superclass are [23, 24, 25, 26] then the returned
            # episode_classes_rel will be [26, 24] which as usual are number relative
            # to the split.
            episode_subclass_ids = sample_class_ids_uniformly(
                                        random_gen,
                                        num_ways,
                                        range(num_superclass_classes))

            (episode_classes_rel,  _) = \
                self.dataset_spec.get_class_ids_from_superclass_subclass_inds(
                self.split, episode_superclass, episode_subclass_ids)
        elif self.use_all_classes:
            episode_classes_rel = np.arange(self.num_classes)  # noqa: E111
        else:  # No type of hierarchy is used. Classes are randomly sampled.
            if self.num_ways is not None:  # noqa: E111
                num_ways = self.num_ways
            else:  # noqa: E111
                num_ways = sample_num_ways_uniformly(
                    random_gen,
                    self.num_filtered_classes,
                    min_ways=self.min_ways,
                    max_ways=self.max_ways_upper_bound)
            # Filtered class IDs relative to the selected split
            ids_rel = [  # noqa: E111
                class_id - self.class_set[0] for class_id in self._filtered_class_set
            ]
            episode_classes_rel = sample_class_ids_uniformly(random_gen, num_ways, ids_rel)  # noqa: E111

        return episode_classes_rel

    def sample_episode_description(self, random_gen: RandomState,):  # noqa: E111
        """Returns the composition of an episode.

        Returns:
          A sequence of `(class_id, num_support, num_query)` tuples, where
            relative `class_id` is an integer in [0, self.num_classes).
        """
        class_ids = self.sample_class_ids(random_gen)
        images_per_class = np.array([
            self.dataset_spec.get_total_images_per_class(
                self.class_set[cid]) for cid in class_ids
        ])

        if self.num_query is not None:
            num_query = self.num_query  # noqa: E111
        else:
            num_query = compute_num_query(  # noqa: E111
                images_per_class,
                max_num_query=self.max_num_query,
                num_support=self.num_support)

        if self.num_support is not None:
            if isinstance(self.num_support, int):  # noqa: E111
                if any(self.num_support + num_query > images_per_class):
                    raise ValueError('Some classes do not have enough examples.')  # noqa: E111
                num_support = self.num_support
            else:  # noqa: E111
                start, end = self.num_support
                if any(end + num_query > images_per_class):
                    raise ValueError('The range provided for uniform sampling of the '  # noqa: E111
                                     'number of support examples per class is not valid: '
                                     'some classes do not have enough examples.')
                num_support = random_gen.randint(low=start, high=end + 1)
            num_support_per_class = [num_support for _ in class_ids]  # noqa: E111
        else:
            num_remaining_per_class = images_per_class - num_query  # noqa: E111
            support_set_size = sample_support_set_size(  # noqa: E111
                random_gen,
                num_remaining_per_class,
                self.max_support_size_contrib_per_class,
                max_support_set_size=self.max_support_set_size)
            num_support_per_class = sample_num_support_per_class(  # noqa: E111
                random_gen,
                images_per_class,
                num_remaining_per_class,
                support_set_size,
                min_log_weight=self.min_log_weight,
                max_log_weight=self.max_log_weight)

        return tuple(
            (class_id, num_support, num_query)
            for class_id, num_support in zip(class_ids, num_support_per_class))

    def compute_chunk_sizes(self):
        """Computes the maximal sizes for the flush, support, and query chunks.

        Sequences of data_lib IDs are padded with dummy IDs to make sure they can be
        batched into episodes of equal sizes.

        The "flush" part of the sequence has a size that is upper-bounded by the
        size of the "support" and "query" parts.

        If variable, the size of the "support" part is in the worst case

            max_support_set_size,

        and the size of the "query" part is in the worst case

            max_ways_upper_bound * max_num_query.

        Returns:
          The sizes of the flush, support, and query chunks.
        """
        if self.num_ways is None:
            max_num_ways = self.max_ways_upper_bound  # noqa: E111
        else:
            max_num_ways = self.num_ways  # noqa: E111

        if self.num_support is None:
            support_chunk_size = self.max_support_set_size  # noqa: E111
        elif isinstance(self.num_support, int):
            support_chunk_size = max_num_ways * self.num_support  # noqa: E111
        else:
            largest_num_support_per_class = self.num_support[1]  # noqa: E111
            support_chunk_size = max_num_ways * largest_num_support_per_class  # noqa: E111

        if self.num_query is None:
            max_num_query = self.max_num_query  # noqa: E111
        else:
            max_num_query = self.num_query  # noqa: E111
        query_chunk_size = max_num_ways * max_num_query

        flush_chunk_size = support_chunk_size + query_chunk_size
        return (flush_chunk_size, support_chunk_size, query_chunk_size)

