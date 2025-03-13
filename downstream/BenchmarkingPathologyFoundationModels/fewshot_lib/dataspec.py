import collections
import json
import os
from typing import Union, List, Any, Dict

import numpy as np
import six
from absl import logging
from six.moves import cPickle as pkl

from fewshot_lib.config import Split


def get_classes(split: Split, classes_per_split: Dict[Split, int]):
    """Gets the sequence of class labels for a split.

    Class id's are returned ordered and without gaps.

    Args:
        split: A Split, the split for which to get classes.
        classes_per_split: Matches each Split to the number of its classes.

    Returns:
        The sequence of classes for the split.

    Raises:
        ValueError: An invalid split was specified.
    """
    num_classes = classes_per_split[split]

    # Find the starting index of classes for the given split.
    if split == Split.TRAIN:
        offset = 0
    elif split == Split.VALID:
        offset = classes_per_split[Split.TRAIN]
    elif split == Split.TEST:
        offset = (
                classes_per_split[Split.TRAIN] +
                classes_per_split[Split.VALID])
    else:
        raise ValueError('Invalid data_lib split.')

    # Get a contiguous range of classes from split.
    return range(offset, offset + num_classes)


def _check_validity_of_restricted_classes_per_split(
            restricted_classes_per_split: Dict[Split, int],
            classes_per_split: Dict[Split, int]):
    """Check the validity of the given restricted_classes_per_split.

    Args:
        restricted_classes_per_split: A dict mapping Split enums to the number of
            classes to restrict to for that split.
        classes_per_split: A dict mapping Split enums to the total available number
            of classes for that split.

    Raises:
        ValueError: if restricted_classes_per_split is invalid.
    """
    for split_enum, num_classes in restricted_classes_per_split.items():
        if split_enum not in [
                Split.TRAIN, Split.VALID,
                Split.TEST
        ]:
            raise ValueError('Invalid key {} in restricted_classes_per_split.'
                             'Valid keys are: Split.TRAIN, '
                             'Split.VALID, and '
                             'Split.TEST'.format(split_enum))
        if num_classes > classes_per_split[split_enum]:
            raise ValueError('restricted_classes_per_split can not specify a '
                             'number of classes greater than the total available '
                             'for that split. Specified {} for split {} but have '
                             'only {} available for that split.'.format(
                                     num_classes, split_enum,
                                     classes_per_split[split_enum]))


def get_total_images_per_class(data_spec, class_id: int = None):
    """Returns the total number of images of a class in a data_spec and pool.

    Args:
        data_spec: A DatasetSpecification, or BiLevelDatasetSpecification.
        class_id: The class whose number of images will be returned. If this is
            None, it is assumed that the data_lib has the same number of images for
            each class.
        pool: A string ('train' or 'test', optional) indicating which example-level
            split to select, if the current data_lib has them.

    Raises:
        ValueError: when
            - no class_id specified and yet there is class imbalance, or
            - no pool specified when there are example-level splits, or
            - pool is specified but there are no example-level splits, or
            - incorrect value for pool.
        RuntimeError: the DatasetSpecification is out of date (missing info).
    """
    if class_id is None:
        if len(set(data_spec.images_per_class.values())) != 1:
            raise ValueError('Not specifying class_id is okay only when all classes'
                             ' have the same number of images')
        class_id = 0

    if class_id not in data_spec.images_per_class:
        raise RuntimeError('The DatasetSpecification should be regenerated, as '
                           'it does not have a non-default value for class_id {} '
                           'in images_per_class.'.format(class_id))
    num_images = data_spec.images_per_class[class_id]

    return num_images

class DatasetSpecification(
        collections.namedtuple('DatasetSpecification', ('name, classes_per_split, images_per_class, '
                               'class_names, path, file_pattern'))):
    """The specification of a data_lib.

        Args:
            name: string, the name of the data_lib.
            classes_per_split: a dict specifying the number of classes allocated to
                each split.
            images_per_class: a dict mapping each class id to its number of images.
                Usually, the number of images is an integer, but if the data_lib has
                'train' and 'test' example-level splits (or "pools"), then it is a dict
                mapping a string (the pool) to an integer indicating how many examples
                are in that pool. E.g., the number of images could be {'train': 5923,
                'test': 980}.
            class_names: a dict mapping each class id to the corresponding class name.
            path: the path to the data_lib's files.
            file_pattern: a string representing the naming pattern for each class's
                file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
                The first gap will be replaced by the class id in both cases, while in
                the latter case the second gap will be replaced with by a shard index,
                or one of 'train', 'valid' or 'test'. This offers support for multiple
                shards of a class' images if a class is too large, that will be merged
                later into a big pool for sampling, as well as different splits that
                will be treated as disjoint pools for sampling the support versus query
                examples of an episode.
    """

    def initialize(self,
                   restricted_classes_per_split: Dict[Split, int] = None):
        """Initializes a DatasetSpecification.

        Args:
            restricted_classes_per_split: A dict that specifies for each split, a
                number to restrict its classes to. This number must be no greater than
                the total number of classes of that split. By default this is None and
                no restrictions are applied (all classes are used).

        Raises:
            ValueError: Invalid file_pattern provided.
        """
        # Check that the file_pattern adheres to one of the allowable forms
        if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
            raise ValueError('file_pattern must be either "{}.tfrecords" or '
                             '"{}_{}.tfrecords" to support shards or splits.')
        if restricted_classes_per_split is not None:
            _check_validity_of_restricted_classes_per_split(
                    restricted_classes_per_split, self.classes_per_split)
            # Apply the restriction.
            for split, restricted_num_classes in restricted_classes_per_split.items():
                self.classes_per_split[split] = restricted_num_classes

    def get_total_images_per_class(self,
                                   class_id: int = None):
        """Returns the total number of images for the specified class.

        Args:
            class_id: The class whose number of images will be returned. If this is
                None, it is assumed that the data_lib has the same number of images for
                each class.
            pool: A string ('train' or 'test', optional) indicating which
                example-level split to select, if the current data_lib has them.

        Raises:
            ValueError: when
                - no class_id specified and yet there is class imbalance, or
                - no pool specified when there are example-level splits, or
                - pool is specified but there are no example-level splits, or
                - incorrect value for pool.
            RuntimeError: the DatasetSpecification is out of date (missing info).
        """
        return get_total_images_per_class(self, class_id)

    def get_classes(self,
                    split: Split):
        """Gets the sequence of class labels for a split.

        Labels are returned ordered and without gaps.

        Args:
            split: A Split, the split for which to get classes.

        Returns:
            The sequence of classes for the split.

        Raises:
            ValueError: An invalid split was specified.
        """
        return get_classes(split, self.classes_per_split)

    def to_dict(self):
        """Returns a dictionary for serialization to JSON.

        Each member is converted_data to an elementary type that can be serialized to
        JSON readily.
        """
        # Start with the dict representation of the namedtuple
        ret_dict = self._asdict()
        # Add the class name for reconstruction when deserialized
        ret_dict['__class__'] = self.__class__.__name__
        # Convert Split enum instances to their name (string)
        ret_dict['classes_per_split'] = {
                split.name: count
                for split, count in six.iteritems(ret_dict['classes_per_split'])
        }
        # Convert binary class names to unicode strings if necessary
        class_names = {}
        for class_id, name in six.iteritems(ret_dict['class_names']):
            if isinstance(name, six.binary_type):
                name = name.decode()
            elif isinstance(name, np.integer):
                name = six.text_type(name)
            class_names[class_id] = name
        ret_dict['class_names'] = class_names
        return ret_dict


class BiLevelDatasetSpecification(
        collections.namedtuple('BiLevelDatasetSpecification',
                               ('name, superclasses_per_split, '
                                'classes_per_superclass, images_per_class, '
                                'superclass_names, class_names, path, '
                                'file_pattern'))):
    """The specification of a data_lib that has a two-level hierarchy.

        Args:
            name: string, the name of the data_lib.
            superclasses_per_split: a dict specifying the number of superclasses
                allocated to each split.
            classes_per_superclass: a dict specifying the number of classes in each
                superclass.
            images_per_class: a dict mapping each class id to its number of images.
            superclass_names: a dict mapping each superclass id to its name.
            class_names: a dict mapping each class id to the corresponding class name.
            path: the path to the data_lib's files.
            file_pattern: a string representing the naming pattern for each class's
                file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
                The first gap will be replaced by the class id in both cases, while in
                the latter case the second gap will be replaced with by a shard index,
                or one of 'train', 'valid' or 'test'. This offers support for multiple
                shards of a class' images if a class is too large, that will be merged
                later into a big pool for sampling, as well as different splits that
                will be treated as disjoint pools for sampling the support versus query
                examples of an episode.
    """

    def initialize(self,
                   restricted_classes_per_split: Union[Split, int] = None):
        """Initializes a DatasetSpecification.

        Args:
            restricted_classes_per_split: A dict that specifies for each split, a
                number to restrict its classes to. This number must be no greater than
                the total number of classes of that split. By default this is None and
                no restrictions are applied (all classes are used).

        Raises:
            ValueError: Invalid file_pattern provided
        """
        # Check that the file_pattern adheres to one of the allowable forms
        if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
            raise ValueError('file_pattern must be either "{}.tfrecords" or '
                             '"{}_{}.tfrecords" to support shards or splits.')
        if restricted_classes_per_split is not None:
            # Create a dict like classes_per_split of DatasetSpecification.
            classes_per_split = {}
            for split in self.superclasses_per_split.keys():
                num_split_classes = self._count_classes_in_superclasses(
                        self.get_superclasses(split))
                classes_per_split[split] = num_split_classes

            _check_validity_of_restricted_classes_per_split(
                    restricted_classes_per_split, classes_per_split)
        # The restriction in this case is applied in get_classes() below.
        self.restricted_classes_per_split = restricted_classes_per_split

    def get_total_images_per_class(self,
                                   class_id: int = None):
        """Returns the total number of images for the specified class.

        Args:
            class_id: The class whose number of images will be returned. If this is
                None, it is assumed that the data_lib has the same number of images for
                each class.
            pool: A string ('train' or 'test', optional) indicating which
                example-level split to select, if the current data_lib has them.

        Raises:
            ValueError: when
                - no class_id specified and yet there is class imbalance, or
                - no pool specified when there are example-level splits, or
                - pool is specified but there are no example-level splits, or
                - incorrect value for pool.
            RuntimeError: the DatasetSpecification is out of date (missing info).
        """
        return get_total_images_per_class(self, class_id)

    def get_superclasses(self, split: Split):
        """Gets the sequence of superclass labels for a split.

        Labels are returned ordered and without gaps.

        Args:
            split: A Split, the split for which to get the superclasses.

        Returns:
            The sequence of superclasses for the split.

        Raises:
            ValueError: An invalid split was specified.
        """
        return get_classes(split, self.superclasses_per_split)

    def _count_classes_in_superclasses(self, superclass_ids: List[int]):
        return sum([
                self.classes_per_superclass[superclass_id]
                for superclass_id in superclass_ids
        ])

    def _get_split_offset(self, split: Split):
        """Returns the starting class id of the contiguous chunk of ids of split.

        Args:
            split: A Split, the split for which to get classes.

        Raises:
            ValueError: Invalid data_lib split.
        """
        if split == Split.TRAIN:
            offset = 0
        elif split == Split.VALID:
            previous_superclasses = range(
                    0, self.superclasses_per_split[Split.TRAIN])
            offset = self._count_classes_in_superclasses(previous_superclasses)
        elif split == Split.TEST:
            previous_superclasses = range(
                    0, self.superclasses_per_split[Split.TRAIN] +
                    self.superclasses_per_split[Split.VALID])
            offset = self._count_classes_in_superclasses(previous_superclasses)
        else:
            raise ValueError('Invalid data_lib split.')
        return offset

    def get_classes(self, split: Split):
        """Gets the sequence of class labels for a split.

        Labels are returned ordered and without gaps.

        Args:
            split: A Split, the split for which to get classes.

        Returns:
            The sequence of classes for the split.
        """
        if not hasattr(self, 'restricted_classes_per_split'):
            self.initialize()
        offset = self._get_split_offset(split)
        if (self.restricted_classes_per_split is not None and
                split in self.restricted_classes_per_split):
            num_split_classes = self.restricted_classes_per_split[split]
        else:
            # No restriction, so include all classes of the given split.
            num_split_classes = self._count_classes_in_superclasses(
                    self.get_superclasses(split))

        return range(offset, offset + num_split_classes)

    def get_class_ids_from_superclass_subclass_inds(self,
                                                    split: Split,
                                                    superclass_id: int,
                                                    class_inds: List[int]):
        """Gets the class ids of a number of classes of a given superclass.

        Args:
            split: A Split, the split for which to get classes.
            superclass_id: An int. The id of a superclass.
            class_inds: A list or sequence of ints. The indices into the classes of
                the superclass superclass_id that we wish to return class id's for.

        Returns:
            rel_class_ids: A list of ints of length equal to that of class_inds. The
                class id's relative to the split (between 0 and num classes in split).
            class_ids: A list of ints of length equal to that of class_inds. The class
                id's relative to the data_lib (between 0 and the total num classes).
        """
        # The number of classes before the start of superclass_id, i.e. the class id
        # of the first class of the given superclass.
        superclass_offset = self._count_classes_in_superclasses(
                range(superclass_id))

        # Absolute class ids (between 0 and the total number of data_lib classes).
        class_ids = [superclass_offset + class_ind for class_ind in class_inds]

        # Relative (between 0 and the total number of classes in the split).
        # This makes the assumption that the class id's are in a contiguous range.
        rel_class_ids = [
                class_id - self._get_split_offset(split) for class_id in class_ids
        ]

        return rel_class_ids, class_ids

    def to_dict(self):
        """Returns a dictionary for serialization to JSON.

        Each member is converted_data to an elementary type that can be serialized to
        JSON readily.
        """
        # Start with the dict representation of the namedtuple
        ret_dict = self._asdict()
        # Add the class name for reconstruction when deserialized
        ret_dict['__class__'] = self.__class__.__name__
        # Convert Split enum instances to their name (string)
        ret_dict['superclasses_per_split'] = {
                split.name: count
                for split, count in six.iteritems(ret_dict['superclasses_per_split'])
        }
        return ret_dict

def as_dataset_spec(dct: Dict[str, Any]):
    """Hook to `json.loads` that builds a DatasetSpecification from a dict.

    Args:
         dct: A dictionary with string keys, corresponding to a JSON file.

    Returns:
        Depending on the '__class__' key of the dictionary, a DatasetSpecification,
        HierarchicalDatasetSpecification, or BiLevelDatasetSpecification. Defaults
        to returning `dct`.
    """
    if '__class__' not in dct:
        return dct

    if dct['__class__'] not in ('DatasetSpecification',
                                'HierarchicalDatasetSpecification',
                                'BiLevelDatasetSpecification'):
        return dct

    def _key_to_int(dct):
        """Returns a new dictionary whith keys converted_data to ints."""
        return {int(key): value for key, value in six.iteritems(dct)}

    def _key_to_split(dct):
        """Returns a new dictionary whith keys converted_data to Split enums."""
        return {
                Split[key]: value for key, value in six.iteritems(dct)
        }

    if dct['__class__'] == 'DatasetSpecification':
        images_per_class = {}
        for class_id, n_images in six.iteritems(dct['images_per_class']):
            # If n_images is a dict, it maps each class ID to a string->int
            # dictionary containing the size of each pool.
            if isinstance(n_images, dict):
                # Convert the number of classes in each pool to int.
                n_images = {
                        pool: int(pool_size) for pool, pool_size in six.iteritems(n_images)
                }
            else:
                n_images = int(n_images)
            images_per_class[int(class_id)] = n_images

        return DatasetSpecification(
                name=dct['name'],
                classes_per_split=_key_to_split(dct['classes_per_split']),
                images_per_class=images_per_class,
                class_names=_key_to_int(dct['class_names']),
                path=dct['path'],
                file_pattern=dct['file_pattern'])

    elif dct['__class__'] == 'BiLevelDatasetSpecification':
        return BiLevelDatasetSpecification(
                name=dct['name'],
                superclasses_per_split=_key_to_split(dct['superclasses_per_split']),
                classes_per_superclass=_key_to_int(dct['classes_per_superclass']),
                images_per_class=_key_to_int(dct['images_per_class']),
                superclass_names=_key_to_int(dct['superclass_names']),
                class_names=_key_to_int(dct['class_names']),
                path=dct['path'],
                file_pattern=dct['file_pattern'])
    else:
        return dct


def load_dataset_spec(dataset_records_path: str, convert_from_pkl: bool = False):
    """Loads data_lib specification from directory containing the data_lib records.

    Newly-generated datasets have the data_lib specification serialized as JSON,
    older ones have it as a .pkl file. If no JSON file is present and
    `convert_from_pkl` is passed, this method will load the .pkl and serialize it
    to JSON.

    Args:
        dataset_records_path: A string, the path to the directory containing
            .tfrecords files and dataset_spec.
        convert_from_pkl: A boolean (False by default), whether to convert a
            dataset_spec.pkl file to JSON.

    Returns:
        A DatasetSpecification, BiLevelDatasetSpecification, or
            HierarchicalDatasetSpecification, depending on the data_lib.

    Raises:
        RuntimeError: If no suitable dataset_spec file is found in directory
            (.json or .pkl depending on `convert_from_pkl`).
    """
    json_path = os.path.join(dataset_records_path, 'dataset_spec.json')
    pkl_path = os.path.join(dataset_records_path, 'dataset_spec.pkl')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data_spec = json.load(f, object_hook=as_dataset_spec)
    elif os.path.exists(pkl_path):
        if convert_from_pkl:
            logging.info('Loading older dataset_spec.pkl to convert it.')
            with open(pkl_path, 'rb') as f:
                data_spec = pkl.load(f)
            with open(json_path, 'w') as f:
                json.dump(data_spec.to_dict(), f, indent=2)
        else:
            raise RuntimeError(
                    'No dataset_spec.json file found in directory %s, but an older '
                    'dataset_spec.pkl was found. You can try to pass '
                    '`convert_from_pkl=True` to convert it, or you may need to run the '
                    'conversion again in order to make sure you have the latest version.'
                    % dataset_records_path)
    else:
        raise RuntimeError('No dataset_spec file found in directory %s' % dataset_records_path)

    # Replace outdated path of where to find the data_lib's records.
    data_spec = data_spec._replace(path=dataset_records_path)
    return data_spec
