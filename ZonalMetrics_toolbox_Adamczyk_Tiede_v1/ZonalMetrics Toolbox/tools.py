# coding=utf-8
import json
from string import ascii_lowercase
import tempfile
import traceback
import sys
import random
import re
from itertools import islice
import os
import datetime

import arcpy

__author__ = u'Adamczyk_Tiede_2015'
__version__ = '20150715_1835'

default_config = {
    'debug': False,
    'debugDir': 'c:/tmp/zonal_metrics_debug',
    'scratchworkspace': "in_memory"
}

cfg = None


def _load_config():
    c = default_config
    config_file_path = os.path.join(os.path.dirname(__file__), 'ZonalMetrics_config.json')
    print config_file_path
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path) as f:
                c2 = json.load(f)
                c.update(c2)
        except (IOError, ValueError):
            traceback.print_exc()
            pass
    return c


def get_config():
    global cfg
    if cfg is None:
        cfg = _load_config()
    return cfg


def get_scratchworkspace():
    return get_config()['scratchworkspace']


def enum(**enums):
    return type('Enum', (), enums)


def _create_query_from_class_list(classField, classList):
    """
    Creates query in form of "classField IN (values, ...)" where values are properly enclosed in apostrophes if it's necessary
    """
    values = [fieldValue.replace("'", "''") for fieldValue in classList]  # escape apostrophes in values
    aps = "'"
    if classField.type in ['Integer', 'SmallInteger', 'Double']:
        aps = ""
    infix = aps + ", " + aps
    return "%(fieldName)s IN (%(aps)s%(values)s%(aps)s)" % {'fieldName': classField.name, 'aps': aps,
                                                            'values': infix.join(values)}


def get_field(feature_class, field_name, on_not_exists=None):
    fields = arcpy.Describe(feature_class).fields
    for field in [f for f in fields if f.name == field_name]:
        return field
    msg = 'Field %s not found' % field_name
    if on_not_exists is None:
        raise FieldNotFoundException(msg)
    else:
        return on_not_exists()


class intersect_analyzed_with_stat_layer(object):
    """
    Creates temporary feature class with is intersection of hexagon layer and analyzed feature class.
    Additionally intersection can be restricted to selected classes.
    This function is made to be used in 'with' statement, so this temporary layer will be cleaned up after exit from 'with' block.
    Use it like this:

    with createHexIntersect(hexOut, inputArea) as hexIntersect:
        SearchCursor(hexIntersect, 'unitID')...
        #do something else
    #here hexIntersect will not exists anymore

    """

    def __init__(self, statistics_output_layer, input_layer, class_field=None, class_list=None, workspace='in_memory'):
        self.statistics_output_layer = statistics_output_layer
        self.input_layer = input_layer
        self.class_field = class_field
        self.class_list = class_list
        self.workspace = workspace
        self.to_cleanup = []

    def __enter__(self):
        statistics_layer_intersection = create_temp_layer_name('stat_intersect')
        statistics_layer_intersect_with_selection = statistics_layer_intersection + '_selected'
        self.to_cleanup += [statistics_layer_intersection, statistics_layer_intersect_with_selection]
        try:
            if self.class_field is not None and self.class_list and len(self.class_list) > 0:
                select_features_from_feature_class(self.class_field, self.class_list, self.input_layer,
                                                   statistics_layer_intersect_with_selection)
            else:
                arcpy.CopyFeatures_management(self.input_layer, statistics_layer_intersect_with_selection)

            log('Creating statistics intersection')
            input_features = [self.statistics_output_layer, statistics_layer_intersect_with_selection]
            arcpy.Intersect_analysis(input_features, statistics_layer_intersection)
            return statistics_layer_intersection
        except Exception, e:
            handleException(e)
            raise e

    def __exit__(self, type, value, traceback):  # @ReservedAssignment
        on_debug(*self.to_cleanup)
        delete_if_exists(*self.to_cleanup)


def is_debug():
    return get_config()['debug']


def setup_debug_mode():
    log('ZonalMetricsTool version: %s' % __version__)
    if is_debug():
        log_debug('Running in debug mode')
        tmp_dir = '/tmp'
        if not os.path.exists(tmp_dir):
            tmp_dir = tempfile.gettempdir()
        debug_dir = os.path.join(tmp_dir, 'zonal_metrics_debug/{:%Y%m%d_%H_%M_%S}'.format(datetime.datetime.now()))
        debug_dir = debug_dir.replace('/', os.sep)
        try:
            os.makedirs(debug_dir)
        except IOError:
            pass
        log_debug('Debug dir: %s' % debug_dir)
        get_config()['debugDir'] = debug_dir
    else:
        log('Debug mode disabled')


def log_debug(message):
    if is_debug():
        log('[DEBUG] %s' % message)


def log(message):
    arcpy.AddMessage(message)
    # see http://support.esri.com/en/knowledgebase/techarticles/detail/35380
    try:
        print message
    except IOError:
        pass
    if not is_debug():
        return


_temp_layer_nb = 0


def create_temp_layer_name(prefix='tempLayer', workspace='in_memory'):
    global _temp_layer_nb
    step_higher = _temp_layer_nb / 26
    step_lower = _temp_layer_nb - step_higher * 26
    step = ascii_lowercase[step_higher] + ascii_lowercase[step_lower]

    for i in xrange(0, 100):  # @UnusedVariable
        name = "%(workspace)s\\%(step)s_%(prefix)s_%(rnd)d" % {'workspace': workspace,
                                                               'step': step,
                                                               'prefix': prefix,
                                                               'rnd': random.randint(0, 1000000)}
        if not arcpy.Exists(name):
            _temp_layer_nb += 1
            _temp_layer_nb %= 676
            return name
    raise Exception('Could not reserve name in 100 tries')


class createTempLayer(object):
    """
    Creates temporary name for use in ArcGIS tools. This function is made to be used in 'with' statement, so this temporary layer created in it will be cleaned up after exit from 'with' block.
    Use it like this:

    with createTempLayer(hexOut, inputArea) as tmpLayer:
        #do something with tmpLayer
    #here tmpLayer will not exists anymore
    """

    def __init__(self, prefix='tempLayer', workspace='in_memory'):
        self.prefix = prefix
        self.workspace = workspace
        self.name = None

    def __enter__(self):
        self.name = create_temp_layer_name(self.prefix, self.workspace)
        return self.name

    def __exit__(self, type, value, traceback):  # @ReservedAssignment
        on_debug(self.name)
        delete_if_exists(self.name)


def on_debug(*feature_classes):
    """

    :type feature_classes: string
    """
    for fc in feature_classes:
        if is_debug() and arcpy.Exists(fc):
            fc_name = fc.partition(os.sep)[2]
            description = arcpy.Describe(fc)
            data_type = getattr(description, 'dataType', False)
            debug_dir = get_config()['debugDir']
            output_name = debug_dir + os.sep + fc_name
            log_debug('Copying {} to {}'.format(fc, output_name))
            if data_type in ['FeatureClass', 'FeatureLayer']:
                arcpy.CopyFeatures_management(fc, output_name)
            elif data_type == 'Table':
                suffix = '' if '.dbf' in output_name else '.dbf'
                arcpy.CopyRows_management(fc, output_name + suffix)
            else:
                log_debug('on_debug - Unhandled dataType %s' % data_type)


class FieldNotFoundException(Exception):
    pass


def delete_if_exists(*arg):
    for fc in arg:
        if arcpy.Exists(fc):
            arcpy.Delete_management(fc)


def handleException(e):
    if is_debug():
        traceback.print_exc()
        # Get the traceback object
    #
    tb = sys.exc_info()[2]
    tbinfo = traceback.format_tb(tb)

    # Concatenate information together concerning the error into a message string
    #
    pymsg = "PYTHON ERRORS:\nTraceback info:\n" + ''.join(tbinfo) + "\nError Info:\n" + str(sys.exc_info()[1])
    msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"

    # Return python error messages for use in script tool or Python Window
    #
    arcpy.AddError(pymsg)
    arcpy.AddError(msgs)

    # Print Python error messages for use in Python / Python Window
    #
    print pymsg + "\n"
    print msgs


class ScriptParameters(object):
    # see http://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
    def __init__(self, parameters, **kwargs):
        for p in parameters:
            setattr(self, p.name, p)
        for key in kwargs:
            setattr(self, key, kwargs[key])


def getParameterValues(parameter):
    values = getattr(parameter, 'values', None)
    if values is not None:
        return values
    if not getattr(parameter, 'multiValue', None):
        return parameter.value
    value_as_text = parameter.valueAsText
    values = []
    while value_as_text:
        value = None
        i = -1
        if value_as_text[0] == "'":
            i = re.search(r'\';|\'$', value_as_text[1:]).end()
            if i == (len(value_as_text) - 1):  # last value
                i = len(value_as_text)
            value = value_as_text[1:i - 1]  # to strip apostrophe
        else:
            i = value_as_text.find(';', 1)
            if i < 0:  # last value without apostrophe
                value = value_as_text
                value_as_text = ''
            else:
                value = value_as_text[:i]
        value_as_text = value_as_text[i:]
        if value_as_text and value_as_text[0] == ';':
            value_as_text = value_as_text[1:]
        if value:
            values.append(value)
    return values


def select_features_from_feature_class(select_field, selected_classes, input_feature_class, output_feature_class):
    where_clause = _create_query_from_class_list(classField=select_field, classList=selected_classes)
    log('Selecting features by clause {clause} from layer {input_feature_class}'.format(clause=where_clause,
                                                                                        input_feature_class=input_feature_class))
    with createTempLayer('__selection') as tmpSel:
        arcpy.MakeFeatureLayer_management(input_feature_class, tmpSel)
        arcpy.SelectLayerByAttribute_management(tmpSel, "NEW_SELECTION", where_clause)
        arcpy.CopyFeatures_management(tmpSel, output_feature_class)


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))
