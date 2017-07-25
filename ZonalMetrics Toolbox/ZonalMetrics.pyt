# coding=utf-8
import math
from collections import Counter, OrderedDict
import random
import sys
import os

import arcpy


# noinspection PyUnresolvedReferences
from arcpy.da import SearchCursor, UpdateCursor, InsertCursor
import itertools

ZONE_AREA_FIELD_NAME = "zone_area"
UNIT_ID_FIELD_NAME = "unitID"

if 'tools' in sys.modules:
    reload(sys.modules['tools'])
from tools import log, is_debug, get_field, intersect_analyzed_with_stat_layer, \
    FieldNotFoundException, ScriptParameters, \
    getParameterValues, handleException, delete_if_exists, createTempLayer, select_features_from_feature_class, \
    setup_debug_mode, enum, create_temp_layer_name, on_debug, get_scratchworkspace, log_debug

__authors_and_citation__ = u'Joanna Adamczyk, Dirk Tiede, ZonalMetrics - a Python toolbox for zonal landscape structure analysis, Computers & Geosciences, Volume 99, February 2017, Pages 91-99, ISSN 0098-3004, http://dx.doi.org/10.1016/j.cageo.2016.11.005'
__license___ = 'GPL-3 GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007'
__version__ = '1.0 for ArcGIS 10.1 or newer'

default_encoding = sys.getdefaultencoding()
# arcpy.env.scratchWorkspace = get_scratchworkspace()


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "ZonalMetrics Tools"
        self.alias = "zonalmetrics"

        # List of tool classes associated with this toolbox
        self.tools = [CreateHexagons,
                      CreatePie,
                      AreaMetrics,
                      DiversityMetricsTool,
                      EdgeMetricsTool,
                      LargestPatchIndex,
                      ContrastMetricsTool,
                      # MetricsGrouping,
                      ConnectanceMetricsTool,
                      ]


class LayerPrepare(object):
    """
    Prepares layer
    """

    def prepare(self, layer):
        raise NotImplementedError()

    def update_row(self, fields, row):
        pass


class EnsureFieldExists(LayerPrepare):
    def __init__(self, field_name, field_type, default_value=None, force_default_value=True):
        LayerPrepare.__init__(self)
        self.field_name = field_name
        self.field_type = field_type
        self.default_value = default_value
        self.field_added = False
        self.force_default_value = force_default_value

    def prepare(self, layer):
        try:
            return get_field(layer, self.field_name)
        except FieldNotFoundException:
            arcpy.AddField_management(layer,
                                      self.field_name,
                                      self.field_type,
                                      "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
            self.field_added = True
        return get_field(layer, self.field_name)

    def update_row(self, fields, row):
        LayerPrepare.update_row(self, fields, row)
        if (self.field_added or self.force_default_value) and self.default_value is not None:
            field_idx = fields[self.field_name]
            row[field_idx] = self.default_value


class EnsureZoneAreaFieldExists(EnsureFieldExists):
    def __init__(self):
        EnsureFieldExists.__init__(self, ZONE_AREA_FIELD_NAME, "DOUBLE", force_default_value=False)

    def update_row(self, fields, row):
        EnsureFieldExists.update_row(self, fields, row)
        if self.field_added:
            geometry_idx = fields['SHAPE@']
            zone_area_idx = fields[ZONE_AREA_FIELD_NAME]
            geometry = row[geometry_idx]
            row[zone_area_idx] = geometry.area


class EnsureUnitIDFieldExists(EnsureFieldExists):
    def __init__(self):
        EnsureFieldExists.__init__(self, UNIT_ID_FIELD_NAME, "LONG", force_default_value=False)

    def update_row(self, fields, row):
        EnsureFieldExists.update_row(self, fields, row)
        if self.field_added:
            unit_id_idx = fields[UNIT_ID_FIELD_NAME]
            fid_idx = fields['FID']
            row[unit_id_idx] = row[fid_idx]


class MetricsCalcTool(object):
    def __init__(self):
        self.category = "ZonalMetrics"
        self.label = "FILL ME IN SUBCLASS"
        self.description = "FILL ME IN SUBCLASS"
        self.canRunInBackground = True
        self._temp_layers = []
        self._temp_layer_nb = 1

    def getParameterInfo(self):
        inputArea = arcpy.Parameter(
            displayName="Input layer",
            name="in_area",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        statLayer = arcpy.Parameter(
            displayName="Statistical layer",
            name="stat_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        out = arcpy.Parameter(
            displayName="Output Feature",
            name="out_feature",
            datatype="Feature Class",
            parameterType="Derived",
            direction="Output")

        out.parameterDependencies = [statLayer.name]
        out.schema.clone = True

        classField = arcpy.Parameter(
            displayName="Class Field",
            name="class_field",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        classField.parameterDependencies = [inputArea.name]

        classList = arcpy.Parameter(
            displayName="Classes",
            name="class_list",
            datatype="String",
            parameterType="Optional",
            direction="Input",
            multiValue=True)
        classList.parameterDependencies = [classField.name]

        return [inputArea, statLayer, classField, classList, out]

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def get_all_classes(self, input_area, class_field_name, map_func=None):
        f = map_func
        if f is None:
            f = lambda x: x
        with SearchCursor(input_area, class_field_name) as cur:
            for row in cur:
                yield f(row[0])

    def _prepareClassListParmeterValues(self, parameter, inputFeature, fieldName, escape=None):
        parameter.filter.type = 'ValueList'
        uniqueValues = OrderedDict.fromkeys(self.get_all_classes(inputFeature, fieldName, map_func=str)).keys()
        if escape is not None:
            uniqueValues = [escape(value) for value in uniqueValues]
        if '' in uniqueValues:
            uniqueValues.remove('')
            parameter.setWarningMessage('Table contains empty class names which were removed from selection');
        parameter.filter.list = sorted(uniqueValues)

    def updateParameters(self, params):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        parameters = ScriptParameters(params)
        class_field_param = parameters.class_field
        class_list_param = parameters.class_list


        if class_field_param.altered:
            try:
                fieldName = class_field_param.valueAsText
                inputFeature = parameters.in_area.value
                selected = getattr(class_list_param, 'values', None)
                if selected:
                    all_classes = self.get_all_classes(inputFeature, fieldName, map_func=str)
                    class_list_param.values = list(
                        set(selected).intersection(set(all_classes)))
                self._prepareClassListParmeterValues(class_list_param, inputFeature, fieldName, escape=None)
            except UnicodeEncodeError:
                # will be handled in updateMessages
                pass

        return

    def _escape(self, value):
        return value.replace('\\\\', '\\').replace("'", "\\'").replace(";", "\;")

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        parameters = ScriptParameters(parameters)
        class_field_param = parameters.class_field
        field_name = class_field_param.valueAsText
        input_feature = parameters.in_area.value

        zonal_param = parameters.in_area
        stat_param = parameters.stat_layer


        if zonal_param.value is None:
            zonal_param.clearMessage()
        else:
            desc = arcpy.Describe(zonal_param.value)
            feature_type = desc.shapeType.lower()
            file_extension = desc.extension
            if not feature_type == "polygon" or not file_extension == "shp":
                zonal_param.setErrorMessage("Only polygon shapefiles allowed")

        if stat_param.value is None:
            stat_param.clearMessage()
        else:
            desc = arcpy.Describe(stat_param.value)
            feature_type = desc.shapeType.lower()
            file_extension = desc.extension
            if not feature_type == "polygon" or not file_extension == "shp":
                stat_param.setErrorMessage("Only polygon shapefiles allowed")



        self.get_all_classes(input_feature, field_name, map_func=str)

        if field_name:
            with SearchCursor(input_feature, field_name) as cur:
                i = 0
                for row in cur:
                    val = row[0]
                    if not isinstance(val, basestring):
                        val = str(val)
                    if len(val.strip()) == 0:
                        msg = 'Column ' + field_name + ' contains empty/blank values in row ' + str(i)
                        class_field_param.setWarningMessage(msg)
                    try:
                        val.encode(default_encoding, 'strict')
                    except UnicodeEncodeError:
                        msg = 'Column ' + field_name + ' contains illegal character in name in row ' + str(
                            i) + '(' + val.encode(default_encoding, 'replace') + ')'
                        class_field_param.setErrorMessage(msg)
                    i += 1

    @staticmethod
    def prepare_stat_layer(layer, *prepares):
        for prepare in prepares:
            assert isinstance(prepare, LayerPrepare)
            prepare.prepare(layer)
        with UpdateCursor(layer, ['*', 'SHAPE@']) as cur:
            fields = OrderedDict(zip(cur.fields, xrange(0, len(cur.fields))))
            for row in cur:
                for prepare in prepares:
                    prepare.update_row(fields, row)
                cur.updateRow(row)

    def create_temp_layer(self, prefix):
        layer = create_temp_layer_name(prefix=prefix)
        self._temp_layer_nb += 1
        self._temp_layers.append(layer)
        return layer

    def on_exit(self):
        on_debug(*self._temp_layers)
        delete_if_exists(*self._temp_layers)


class MetricsGrouping(MetricsCalcTool):
    def __init__(self):
        MetricsCalcTool.__init__(self)
        self.category = "Metrics"
        self.label = "Calculate multiple metrics"

        subcls = MetricsCalcTool.__subclasses__()
        subcls.remove(ContrastMetricsTool)
        subcls.remove(MetricsGrouping)
        subcls.remove(ConnectanceMetricsTool)

        self.tools = [t() for t in subcls]

        self.description = "Calculates selected metrics:<ul>"
        for t in self.tools:
            self.description += "<li>" + t.label + "</li>"
        self.description += "</ul>"

        self.canRunInBackground = True

    def getParameterInfo(self):
        params = MetricsCalcTool.getParameterInfo(self)
        toolsSelection = arcpy.Parameter(
            displayName="Tools to be run",
            name="selected_tools",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        toolsSelection.filter.type = 'ValueList'
        toolsSelection.filter.list = sorted([tool.label for tool in self.tools])
        return [toolsSelection] + params

    def execute(self, parameters, messages):
        setup_debug_mode()
        params = ScriptParameters(parameters)
        selected_tools = params.selected_tools.values
        for tool in self.tools:
            if tool.label not in selected_tools:
                continue
            tool_parameters = parameters[1:]
            log('\n\t---')
            log('\tRunning tool %s' % tool.label)
            log('\t---')
            tool.execute(tool_parameters, messages)
        return

        # subclasses


AreaCalcMethod = enum(CUT_TO_STAT=0, OVERLAP=1, CENTROID=2)
AreaCalcMethod_descriptions = {AreaCalcMethod.CUT_TO_STAT: 'Cut patches',
                               AreaCalcMethod.OVERLAP: 'Select overlapping patches',
                               AreaCalcMethod.CENTROID: 'Select by centroids'}


class AbstractAreaMetricsCalcTool(MetricsCalcTool):
    input_layer = None
    statistics_layer = None
    class_field = None
    selected_classes = None
    input_parameters = None
    area_analysis_method = AreaCalcMethod.CUT_TO_STAT
    process_only_selected = False
    to_cleanup = []
    allow_percent_gt_100 = False

    def __init__(self):
        super(AbstractAreaMetricsCalcTool, self).__init__()

    def getParameterInfo(self):
        params = MetricsCalcTool.getParameterInfo(self)
        area_analysis_method_param = arcpy.Parameter(
            displayName="Area analysis method",
            name="area_analysis_method",
            datatype="String",
            parameterType="Required",
            direction="Input",
            multiValue=False)
        area_analysis_method_param.filter.type = 'ValueList'
        area_analysis_method_param.filter.list = AreaCalcMethod_descriptions.values()
        area_analysis_method_param.value = area_analysis_method_param.filter.list[0]
        return params + [area_analysis_method_param]

    def _parse_parameters(self, parameters):
        self.input_parameters = ScriptParameters(parameters)
        self.input_layer = self.input_parameters.in_area.valueAsText
        self.statistics_layer = self.input_parameters.stat_layer.valueAsText
        class_field_name = self.input_parameters.class_field.valueAsText
        if class_field_name:
            self.class_field = get_field(self.input_layer, class_field_name)
            self.selected_classes = getParameterValues(self.input_parameters.class_list)
        area_analysis_method_value = self.input_parameters.area_analysis_method.valueAsText
        self.area_analysis_method = self.input_parameters.area_analysis_method.filter.list.index(
            area_analysis_method_value)

    def execute(self, parameters, messages):
        setup_debug_mode()
        self._parse_parameters(parameters)

        try:
            self.prepare_stat_layer(self.statistics_layer,
                                    EnsureUnitIDFieldExists(), )
            self.prepare()
            self.process_layer_to_analyse()
        except Exception, e:
            handleException(e)
            raise e
        finally:
            self.cleanup()

    def prepare(self):
        pass

    def update_statistic_row(self, statistic_layer_row, stat_fields, patches_cursor):
        raise NotImplemented()

    def dump_selected_hexagons(self, layer_to_analyse_hids):
        if is_debug():
            selected_hexes = create_temp_layer_name('a_03_selected_hexes')
            self.to_cleanup.append(selected_hexes)
            selected_hexes_out = create_temp_layer_name('a_04_selected_hexes')
            self.to_cleanup.append(selected_hexes_out)

            hids = []
            with SearchCursor(layer_to_analyse_hids, ['stat_id']) as c:
                for row in c:
                    hids.append(str(row[0]))
            arcpy.MakeFeatureLayer_management(self.statistics_layer, selected_hexes)

            where_hids = " \"unitID\" in (%s)" % ", ".join(hids)
            arcpy.SelectLayerByAttribute_management(selected_hexes, "ADD_TO_SELECTION", where_hids)

            arcpy.CopyFeatures_management(selected_hexes, selected_hexes_out)

            on_debug(selected_hexes, selected_hexes_out)

    def _get_input_filtered_to_selection(self):
        log('Selecting features from input layer')
        input_selected = create_temp_layer_name('a_01_1_input_selected')
        self.to_cleanup.append(input_selected)
        if self.class_field is not None and self.selected_classes and len(self.selected_classes) > 0:
            select_features_from_feature_class(self.class_field, self.selected_classes, self.input_layer,
                                               input_selected)
        else:
            arcpy.CopyFeatures_management(self.input_layer, input_selected)
        on_debug(input_selected)
        return input_selected

    def process_layer_to_analyse(self):
        log('Creating intersection of statistics with input layer')

        input_selected = self._get_input_filtered_to_selection()

        layer_to_analyse = create_temp_layer_name('a_02_layer_to_analyse')
        self.to_cleanup.append(layer_to_analyse)

        layer_to_analyse_sorted = create_temp_layer_name('a_03_layer_to_analyse_sorted')
        self.to_cleanup.append(layer_to_analyse_sorted)

        # name of field with FID of found patch in patch_search_layer
        patch_search_field = None
        # name of field with FID of found patch in stat_search_layer
        patch_stat_search_field = None
        # name of field with id of statistic patch (i.e hexagon)
        stat_field = None
        # layer where to search for found hexagons and matched patches
        stat_search_layer = layer_to_analyse_sorted
        # layer where to search for patches to analyse
        patch_search_layer = None

        log('Area analysis method: {}, {}'.format(self.area_analysis_method,
                                                  AreaCalcMethod_descriptions[self.area_analysis_method]))

        if self.area_analysis_method == AreaCalcMethod.CUT_TO_STAT:
            arcpy.Intersect_analysis([self.statistics_layer, input_selected], layer_to_analyse)

##
            arcpy.AddField_management(layer_to_analyse, 'sortFID', "LONG")
            desc = arcpy.Describe(layer_to_analyse)
            FIDfield = desc.OIDFieldName
            arcpy.CalculateField_management(layer_to_analyse, 'sortFID',
                                    "!" + FIDfield + "!",
                                    "PYTHON_9.3")
            patch_stat_search_field = 'sortFID'

##
            #patch_stat_search_field = 'FID'#

            patch_search_field = 'FID'
            stat_field = UNIT_ID_FIELD_NAME
            stat_search_layer = layer_to_analyse_sorted
            patch_search_layer = layer_to_analyse
            self.allow_percent_gt_100 = False

        elif self.area_analysis_method == AreaCalcMethod.OVERLAP:
            match_option = 'INTERSECT'
            log('Joining {} with {} using {}'.format(self.statistics_layer, input_selected, match_option))
            arcpy.SpatialJoin_analysis(self.statistics_layer, input_selected, layer_to_analyse,
                                       join_operation='JOIN_ONE_TO_MANY', join_type='KEEP_COMMON',
                                       match_option=match_option)

            patch_stat_search_field = 'JOIN_FID'
            patch_search_field = 'FID'
            stat_field = UNIT_ID_FIELD_NAME
            stat_search_layer = layer_to_analyse_sorted
            patch_search_layer = input_selected
            self.allow_percent_gt_100 = True

        elif self.area_analysis_method == AreaCalcMethod.CENTROID:
            exploded_polygons = create_temp_layer_name('a_05_singleparts')
            centroids = create_temp_layer_name('a_06_centroids')
            self.to_cleanup.append(exploded_polygons)
            self.to_cleanup.append(centroids)

            arcpy.MultipartToSinglepart_management(input_selected, exploded_polygons)
            arcpy.FeatureToPoint_management(exploded_polygons, centroids)
            on_debug(exploded_polygons, centroids)

            match_option = 'INTERSECT'
            log('Joining {} with {} using {}'.format(self.statistics_layer, centroids, match_option))
            arcpy.SpatialJoin_analysis(self.statistics_layer, centroids, layer_to_analyse,
                                       join_operation='JOIN_ONE_TO_MANY', join_type='KEEP_COMMON',
                                       match_option=match_option)
            patch_stat_search_field = 'ORIG_FID'
            patch_search_field = 'FID'
            stat_field = UNIT_ID_FIELD_NAME
            stat_search_layer = layer_to_analyse_sorted
            patch_search_layer = input_selected
            self.allow_percent_gt_100 = True

        log('Sorting {} by fields {}, {}'.format(layer_to_analyse, stat_field, patch_stat_search_field))
        arcpy.Sort_management(layer_to_analyse, layer_to_analyse_sorted,
                [[stat_field, 'ASCENDING'], [patch_stat_search_field, 'ASCENDING'] ])


        on_debug(layer_to_analyse, layer_to_analyse_sorted)
        log(
            """patch_stat_search_field: {},
            stat_field: {},
            stat_search_layer: {},
            patch_search_layer: {},
            patch_search_field: {}""".format(
                patch_stat_search_field,
                stat_field,
                stat_search_layer,
                patch_search_layer,
                patch_search_field))

        def update_stats(stats_where_clause=None):
            # log(str(w_clause))
            with UpdateCursor(self.statistics_layer, [stat_field, '*', 'SHAPE@AREA', 'SHAPE@LENGTH', 'SHAPE@XY'],
                              where_clause=stats_where_clause) as stat_update_cursor:
                stat_fields = stat_update_cursor.fields
                for statistic_layer_row in stat_update_cursor:
                    arcpy.SetProgressorPosition()
                    stat_id = statistic_layer_row[0]
                    patch_w_clause = '{field} = {value}'.format(
                        field=arcpy.AddFieldDelimiters(stat_search_layer, stat_field),
                        value=stat_id)
                    arcpy.AddMessage(patch_w_clause)
                    patches_ids = [str(row[1]) for row in
                                   SearchCursor(stat_search_layer, [stat_field, patch_stat_search_field],
                                                where_clause=patch_w_clause)]
                    returned_row = None
                    if patches_ids:
                        log('stat_id: {sid},\n    patches ids: {pids}'.format(sid=stat_id, pids=patches_ids))
                        patch_w_clause = '{field} IN ({values})'.format(
                            field=arcpy.AddFieldDelimiters(patch_search_layer, patch_search_field),
                            values=','.join(patches_ids))
                        with SearchCursor(patch_search_layer,
                                          [patch_search_field, '*', 'SHAPE@AREA', 'SHAPE@LENGTH', 'SHAPE@XY'],
                                          where_clause=patch_w_clause) as patches_cursor:
                            returned_row = self.update_statistic_row(statistic_layer_row, stat_fields, patches_cursor)
                    else:
                        returned_row = self.update_statistic_row(statistic_layer_row, stat_fields, [])

                    if returned_row:
                        stat_update_cursor.updateRow(returned_row)

        msg = 'Start of area calculation'
        log(msg)
        if self.process_only_selected:
            arcpy.SetProgressor('default ', msg, 0, 100, 1)
            for stat_ids in _get_distinct_field_values(stat_search_layer, stat_field):
                ids = ', '.join([str(sid) for sid in stat_ids])
                stats_w_clause = '{field} IN ({values})'.format(
                    field=arcpy.AddFieldDelimiters(stat_search_layer, stat_field),
                    values=ids)
                update_stats(stats_w_clause)
        else:
            arcpy.SetProgressor('step', msg, 0, int(arcpy.GetCount_management(self.statistics_layer).getOutput(0)), 1)
            update_stats()
        log('Finished area calculation')
        arcpy.ResetProgressor()
        return None

    def cleanup(self):
        delete_if_exists(self.to_cleanup)

    def prepare_stat_field(self, name, field_type):
        def add_field():
            arcpy.AddField_management(self.statistics_layer, name, field_type, '', '', '', '', "NULLABLE",
                                      "NON_REQUIRED", '')
            return get_field(self.statistics_layer, name)

        return get_field(self.statistics_layer, name, on_not_exists=add_field)


def _get_distinct_field_values(stat_search_layer, stat_field):
    max_records = 50
    recs = set()
    with SearchCursor(stat_search_layer, [stat_field]) as sc:
        for row in sc:
            recs.add(row[0])
            if len(recs) == max_records:
                yield recs
                recs = set()
        if len(recs) > 0:
            yield recs


class AreaMetrics(AbstractAreaMetricsCalcTool):
    fields = {}
    _classes_template = {}

    def __init__(self):
        AbstractAreaMetricsCalcTool.__init__(self)
        self.label = "Area metrics"
        self.description = '''Calculates area metrics'''

    def prepare(self):
        classes = self.selected_classes
        if not self.selected_classes:
            classes = set(self.get_all_classes(self.input_layer, self.class_field.name))

        for cl_raw in classes:
            cl = str(cl_raw).strip("'")
            area_field_name = arcpy.ValidateFieldName('ca%s' % (cl,))[:10]
            count_field_name = arcpy.ValidateFieldName('npc%s' % (cl,))[:10]
            percent_field_name = arcpy.ValidateFieldName('pz%s' % (cl,))[:10]
            self.fields[cl] = {'area': area_field_name, 'count': count_field_name, 'percent': percent_field_name}
            self._classes_template[cl] = 0
        for _, field_names in self.fields.items():
            for field_name in field_names.values():
                self.prepare_stat_field(field_name, 'DOUBLE')
        self.prepare_stat_field(ZONE_AREA_FIELD_NAME, 'DOUBLE')

    def update_statistic_row(self, statistic_layer_row, statistic_layer_fields, patches_cursor):
        stat_area_field_idx = statistic_layer_fields.index('SHAPE@AREA')
        zone_area_field_idx = statistic_layer_fields.index(ZONE_AREA_FIELD_NAME)

        patches_area = Counter(self._classes_template)
        patches_count = Counter(self._classes_template)

        if patches_cursor:
            class_field_idx = patches_cursor.fields.index(self.class_field.name)
            area_field_idx = patches_cursor.fields.index('SHAPE@AREA')
            for patch in patches_cursor:
                class_name = str(patch[class_field_idx])
                patches_area[class_name] += patch[area_field_idx]
                patches_count[class_name] += 1

        for cl_raw in patches_count.keys():
            cl = str(cl_raw)
            this_class_field_names = self.fields[cl]
            area_field_name = this_class_field_names['area']
            count_field_name = this_class_field_names['count']
            percent_field_name = this_class_field_names['percent']
            patches_area_in_hex = patches_area[cl]
            stat_area = statistic_layer_row[stat_area_field_idx]
            statistic_layer_row[statistic_layer_fields.index(area_field_name)] = patches_area_in_hex
            statistic_layer_row[statistic_layer_fields.index(count_field_name)] = patches_count[cl]
            percent = round((patches_area_in_hex / stat_area) * 100, 3)
            if not self.allow_percent_gt_100 and percent > 100:
                log('Warning: percent of area bigger than 100 for row: %s' % str(statistic_layer_row))
            statistic_layer_row[statistic_layer_fields.index(percent_field_name)] = percent

        statistic_layer_row[zone_area_field_idx] = statistic_layer_row[stat_area_field_idx]
        return statistic_layer_row


class LargestPatchIndex(AbstractAreaMetricsCalcTool):
    lpi_field = None
    lpi_class_field = None

    def __init__(self):
        AbstractAreaMetricsCalcTool.__init__(self)
        self.label = "Area Metrics - Largest Patch Index"
        self.description = '''Looks for patch covering the largest area within the statistical zone,
        calculates the area of this patch (LPI) and identifies the class this patch belongs to.
        If proportion of land cover within the statistical zone is considered,
        all patches of considered classes should be dissolved.'''
        self.process_only_selected = False

    def getParameterInfo(self):
        params = AbstractAreaMetricsCalcTool.getParameterInfo(self)
        merge_same_class_patches_param = arcpy.Parameter(
            displayName="Merge patches of the same class",
            name="merge_same_class_patches",
            datatype="Boolean",
            parameterType="Required",
            direction="Input",
            multiValue=False)
        merge_same_class_patches_param.value = False
        params.insert(-2, merge_same_class_patches_param)
        return params

    def _parse_parameters(self, parameters):
        AbstractAreaMetricsCalcTool._parse_parameters(self, parameters)
        self.merge_same_class_patches = self.input_parameters.merge_same_class_patches.value
        log('merge_same_class_patches=%s' % self.merge_same_class_patches)

    def _get_input_filtered_to_selection(self):
        input_filtered = AbstractAreaMetricsCalcTool._get_input_filtered_to_selection(self)

        input_merged = create_temp_layer_name('a_01_2_input_merged')
        self.to_cleanup.append(input_merged)

        if self.merge_same_class_patches:
            arcpy.Dissolve_management(in_features=input_filtered,
                                      out_feature_class=input_merged,
                                      dissolve_field=self.input_parameters.class_field.valueAsText,
                                      multi_part=False
                                      )
        else:
            arcpy.CopyFeatures_management(input_filtered, input_merged)
        on_debug(input_merged)

        return input_merged

    def prepare(self):
        self.lpi_field = self.prepare_stat_field('lpi', 'DOUBLE')
        self.lpi_class_field = self.prepare_stat_field('lpi_class', 'STRING')

    def update_statistic_row(self, statistic_layer_row, statistic_layer_fields, patches_cursor):
        stat_area_field_idx = statistic_layer_fields.index('SHAPE@AREA')
        lpi_field_idx = statistic_layer_fields.index(self.lpi_field.name)
        lpi_class_field_idx = statistic_layer_fields.index(self.lpi_class_field.name)

        statistic_layer_row[lpi_field_idx] = 0
        statistic_layer_row[lpi_class_field_idx] = ''

        if not patches_cursor:
            return statistic_layer_row

        area_field_idx = patches_cursor.fields.index('SHAPE@AREA')
        class_field_idx = patches_cursor.fields.index(self.class_field.name)

        largest_patch = None

        for patch in patches_cursor:
            patch_area = patch[area_field_idx]
            if largest_patch is None or patch_area > largest_patch[area_field_idx]:
                largest_patch = patch
        if largest_patch is None:
            return statistic_layer_row

        statistic_layer_row[lpi_field_idx] = round(
            largest_patch[area_field_idx] / statistic_layer_row[stat_area_field_idx] * 100, 3)
        statistic_layer_row[lpi_class_field_idx] = largest_patch[class_field_idx]

        return statistic_layer_row


class DiversityMetricsTool(MetricsCalcTool):
    diversity_field_name = 'shdi'

    def __init__(self):
        MetricsCalcTool.__init__(self)
        self.label = "Diversity Metrics"
        self.description = '''Calculates Shannon diversity index (SHDI) per zone'''

    @staticmethod
    def _get_class_area_for_zone(input_stat_layer_intersect, unit_id, class_field_name):
        class_area_counter = Counter()
        where = "%s=%s" % (UNIT_ID_FIELD_NAME, unit_id)
        with SearchCursor(input_stat_layer_intersect, (class_field_name, "SHAPE@AREA"), where) as cur2:
            for className, area in cur2:
                class_area_counter[className] += area
        return class_area_counter

    @staticmethod
    def _calculate_shdi(class_areas):
        shdi_raw = 0

        classes_areas = [area for area in class_areas.values() if area > 0]
        class_area_per_zone = sum(classes_areas)

        for class_area in classes_areas:
            shdi_raw += class_area / class_area_per_zone * math.log(class_area / class_area_per_zone)
        shdi = -shdi_raw
        if is_debug():
            log_debug('shdi=%f, zone_area=%d, classes_areas=%s' % (shdi, class_area_per_zone, classes_areas))
        return shdi

    def execute(self, parameters, messages):
        setup_debug_mode()
        input_parameters = ScriptParameters(parameters)
        input_area = input_parameters.in_area.valueAsText
        stat_layer = input_parameters.stat_layer.valueAsText
        class_field_name = input_parameters.class_field.valueAsText
        class_field = get_field(input_area, class_field_name)
        class_list = getParameterValues(input_parameters.class_list)
        self.prepare_stat_layer(stat_layer,
                                EnsureUnitIDFieldExists(),
                                EnsureFieldExists(
                                    field_name=self.diversity_field_name,
                                    field_type='DOUBLE',
                                    default_value=0
                                ),
                                EnsureZoneAreaFieldExists(),

                                )

        msg = "Counting diversity of classes in hexagons"
        log(msg)
        arcpy.SetProgressor("step", msg, 0, int(arcpy.GetCount_management(stat_layer).getOutput(0)), 1)

        fields = [UNIT_ID_FIELD_NAME, "SHAPE@", ZONE_AREA_FIELD_NAME, self.diversity_field_name]
        with intersect_analyzed_with_stat_layer(stat_layer, input_area,
                                                class_field=class_field,
                                                class_list=class_list) as input_stat_layer_intersect, \
                UpdateCursor(stat_layer, fields) as cur:
            for row in cur:
                arcpy.SetProgressorPosition()
                try:
                    zone_area = row[2]
                    if zone_area == 0:
                        continue
                    class_area_counter = self._get_class_area_for_zone(input_stat_layer_intersect, row[0],
                                                                       class_field_name)
                    row[3] = self._calculate_shdi(class_area_counter)
                    cur.updateRow(row)
                except Exception, e:
                    arcpy.AddError('Error: ' + str(e))
        arcpy.ResetProgressor()


class EdgeMetricsTool(MetricsCalcTool):
    density_area = arcpy.Parameter(
        displayName="Edge Density calculation area (ha)",
        name="density_area",
        datatype="Long",
        parameterType="Required",
        direction="Input")
    density_area.value = 1000

    edge_length_field = 'tc_edge'
    edge_density_field = 'ed'

    def __init__(self):
        MetricsCalcTool.__init__(self)
        self.label = 'Edge Metrics'
        self.description = 'Calculates Class Edge lenght (TcE) for all patches of selected classes within the statistical zones'

    def getParameterInfo(self):
        params = MetricsCalcTool.getParameterInfo(self)
        params.append(self.density_area)
        return params

    def execute(self, parameters, messages):
        setup_debug_mode()
        input_parameters = ScriptParameters(parameters)
        input_area = input_parameters.in_area.valueAsText
        stat_layer = input_parameters.stat_layer.valueAsText
        classFieldName = input_parameters.class_field.valueAsText
        classField = get_field(input_area, classFieldName)
        classList = getParameterValues(input_parameters.class_list)

        density_area_ha = int(input_parameters.density_area.valueAsText)  # in ha
        density_area = density_area_ha * 10000  # 1 ha = 10000 square meters

        self.prepare_stat_layer(stat_layer,
                                EnsureUnitIDFieldExists(),
                                EnsureZoneAreaFieldExists(),
                                EnsureFieldExists(self.edge_length_field, "DOUBLE", 0),
                                EnsureFieldExists(self.edge_density_field, "DOUBLE", 0),
                                )

        msg = "Counting edges lengths"
        log(msg)
        arcpy.SetProgressor("step", msg, 0, int(arcpy.GetCount_management(stat_layer).getOutput(0)), 1)

        edgesLayerName = "in_memory\\inputEdges%d" % random.randint(0, 1000000)
        edgesLayerNameLyr = edgesLayerName + 'Layer'
        try:
            arcpy.FeatureToLine_management(input_area, edgesLayerName)
            arcpy.MakeFeatureLayer_management(edgesLayerName, edgesLayerNameLyr)

            stat_layer_fields = [UNIT_ID_FIELD_NAME,
                                 self.edge_length_field,
                                 ZONE_AREA_FIELD_NAME,
                                 "SHAPE@AREA",
                                 self.edge_density_field]
            with intersect_analyzed_with_stat_layer(stat_layer, edgesLayerNameLyr, class_field=classField,
                                                    class_list=classList) as edgesIntersection, \
                    UpdateCursor(stat_layer, stat_layer_fields) as cur:
                for row in cur:
                    arcpy.SetProgressorPosition()
                    try:
                        where = "%s=%s" % (UNIT_ID_FIELD_NAME, row[0])
                        with SearchCursor(edgesIntersection, ("SHAPE@LENGTH",), where) as cur2:
                            edges_length = sum(edges_row[0] for edges_row in cur2)
                            row[1] = edges_length
                            zone_area = row[2]
                            row[4] = edges_length * density_area / zone_area
                        cur.updateRow(row)
                    except Exception, e:
                        arcpy.AddError('Error: ' + str(e))
        finally:
            # cleanup
            arcpy.Delete_management(edgesLayerName)
            arcpy.Delete_management(edgesLayerNameLyr)
        arcpy.ResetProgressor()


class ContrastMetricsTool(MetricsCalcTool):
    def __init__(self):
        MetricsCalcTool.__init__(self)
        self.label = "Contrast Metrics"
        self.description = '''for analyzed class calculates edge length of shared boundaries with selected contrast classes.'''

    def getParameterInfo(self):
        params = MetricsCalcTool.getParameterInfo(self)
        parameters = ScriptParameters(params)
        parameters.class_list.parameterType = "Required"
        parameters.class_list.displayName = "Contrast classes"

        analyzedClass = arcpy.Parameter(
            displayName="Analyzed class",
            name="analyzed_class",
            datatype="String",
            parameterType="Required",
            direction="Input",
        )
        analyzedClass.parameterDependencies = [parameters.in_area.name]
        analyzedClass.filter.type = parameters.class_list.filter.type
        analyzedClass.filter.list = list(parameters.class_list.filter.list)

        params.insert(params.index(parameters.class_list), analyzedClass)
        return params

    def updateParameters(self, params):
        MetricsCalcTool.updateParameters(self, params)
        parameters = ScriptParameters(params)
        analyzedClass = parameters.analyzed_class
        if parameters.class_field.altered:
            analyzedClass.filter.list = list(parameters.class_list.filter.list)

    def execute(self, parameters, messages):
        setup_debug_mode()
        input_parameters = ScriptParameters(parameters)
        input_area = input_parameters.in_area.valueAsText
        stats_out = input_parameters.stat_layer.valueAsText
        classFieldName = input_parameters.class_field.valueAsText
        classField = get_field(input_area, classFieldName)
        analyzed_class = getParameterValues(input_parameters.analyzed_class)

        classList = getParameterValues(input_parameters.class_list)
        if not classList or len(classList) == 0:
            classList = self.get_all_classes(input_area, classFieldName)
        classList = map(str, classList)

        self.prepare_stat_layer(stats_out,
                                EnsureUnitIDFieldExists(), )

        analyzedClassEdgeLength = "el_a_class"
        fields = [analyzedClassEdgeLength]
        fieldNames = {}
        for cl in classList:
            sharedEdgeLengthFieldName = arcpy.ValidateFieldName('el%s' % (cl))[:10]
            contrastIndexFieldName = arcpy.ValidateFieldName('cce%s' % (cl))[:10]
            fields += [sharedEdgeLengthFieldName, contrastIndexFieldName]
            fieldNames[cl] = {'edgeLength': sharedEdgeLengthFieldName,
                              'contrastIndex': contrastIndexFieldName}
        try:
            for fieldName in fields:
                log('Adding field %s to %s' % (fieldName, stats_out))
                arcpy.AddField_management(stats_out, fieldName, "DOUBLE", "", "", "", "", "NULLABLE", "NON_REQUIRED",
                                          "")

            msg = "Counting Class Edge Contrast Index"
            log(msg)
            arcpy.SetProgressor("step", msg, 0, int(arcpy.GetCount_management(stats_out).getOutput(0)) * 2, 1)

            with createTempLayer('analyzed_class') as analyzedInputClass, \
                    createTempLayer('analyzedInputClassEdges') as analyzedInputClassEdges, \
                    createTempLayer('analyzedInputClassBuffered') as analyzedInputClassBuffered, \
                    createTempLayer('boundaryInputClasses') as boundaryInputClasses, \
                    createTempLayer('boundaryInputClassesEdges') as boundaryInputClassesEdges, \
                    createTempLayer('boundaryInputClassesEdgesClipped') as boundaryInputClassesEdgesClipped:

                select_features_from_feature_class(classField, [analyzed_class], input_area, analyzedInputClass)
                arcpy.FeatureToLine_management(analyzedInputClass, analyzedInputClassEdges)
                arcpy.Buffer_analysis(analyzedInputClassEdges, analyzedInputClassBuffered,
                                      buffer_distance_or_field='0.05 meter', line_end_type='FLAT')

                select_features_from_feature_class(classField, classList, input_area, boundaryInputClasses)
                arcpy.Snap_edit(analyzedInputClass, [[boundaryInputClasses, 'EDGE', '1 meter']])
                arcpy.FeatureToLine_management(boundaryInputClasses, boundaryInputClassesEdges)
                arcpy.Clip_analysis(in_features=boundaryInputClassesEdges, clip_features=analyzedInputClassBuffered,
                                    out_feature_class=boundaryInputClassesEdgesClipped)

                # calculate total edge length
                # TODO - integrate with Edge Density tool
                with intersect_analyzed_with_stat_layer(stats_out,
                                                        analyzedInputClassEdges) as edgesIntersection, UpdateCursor(
                    stats_out, (UNIT_ID_FIELD_NAME, analyzedClassEdgeLength)) as cur:
                    for row in cur:
                        arcpy.SetProgressorPosition()
                        try:
                            where = "%s=%s" % (UNIT_ID_FIELD_NAME, row[0])
                            with SearchCursor(edgesIntersection, ("SHAPE@LENGTH",), where) as cur2:
                                edgesLength = sum(row[0] for row in cur2)
                                row[1] = edgesLength
                            cur.updateRow(row)
                        except Exception, e:
                            handleException(e)

                with intersect_analyzed_with_stat_layer(stats_out,
                                                        boundaryInputClassesEdgesClipped) as edgesIntersection, UpdateCursor(
                    stats_out, [UNIT_ID_FIELD_NAME, 'SHAPE@'] + fields) as cur:
                    cursorFields = cur.fields
                    analyzedClassEdgeLengthPos = cursorFields.index(analyzedClassEdgeLength)
                    for row in cur:
                        arcpy.SetProgressorPosition()
                        try:
                            where = "%s=%s" % (UNIT_ID_FIELD_NAME, row[0])
                            with SearchCursor(edgesIntersection, ("SHAPE@LENGTH", classFieldName), where) as cur2:
                                cnt = Counter()
                                for row2 in cur2:
                                    className = str(row2[1])
                                    edgeLength = row2[0]
                                    cnt[className] += edgeLength
                                for className, edgeLength in cnt.items():
                                    fieldName = fieldNames[className]
                                    lengthPos = cursorFields.index(fieldName['edgeLength'])
                                    contrastIndexPos = cursorFields.index(fieldName['contrastIndex'])
                                    row[lengthPos] = edgeLength
                                    row[contrastIndexPos] = edgeLength / row[analyzedClassEdgeLengthPos] * 100
                            cur.updateRow(row)
                        except Exception, e:
                            handleException(e)
            arcpy.ResetProgressor()
        except Exception, e:
            arcpy.DeleteField_management(stats_out, fields)
            handleException(e)


class ConnectanceMetricsTool(MetricsCalcTool):
    PATCH_AREA_PERCENT_FIELD_NAME = 'ci_pp'  # percentage of patch area
    PATCH_AREA_FIELD_NAME = 'ci_pa'  # patch area in range of connection
    CONNECTION_AREA_PERCENTAGE_FIELD_NAME = 'ci_cp'  # percentage of connection area to hex area
    CONNECTION_AREA_FIELD_NAME = 'ci_ca'  # connection area
    NUMBER_OF_PATCHES_FIELD_NAME = 'ci_np'  # number of connected patches (by distinct id)

    def __init__(self):
        MetricsCalcTool.__init__(self)

        self.label = "Connectance Metrics"
        self.description = '''Calculates a Connectance Index and Connection area layer within a given distance between selected classes.
        The detailed explanation can be found here https://docs.google.com/drawings/d/1dQ2WId9RD5sfMeJvY63-uVbrSNz228fHe3JDkYe3-6o/edit?usp=sharing<br/>
        Outputs:
<ul>
   <li> <em>ci_np</em> - number of distinct connected patches (by FID) </li>
   <li> <em>ci_pa</em> - patch area within range of connection </li>
   <li> <em>ci_pp</em> - percentage of patch area within range of connection to statistical zone area</li>
   <li> <em>ci_ca</em> - area of connection zone between patches</li>
   <li> <em>ci_cp</em> - percentage of connection zone between patches to statistical zone area</li>
</ul>'''

    def getParameterInfo(self):
        params = MetricsCalcTool.getParameterInfo(self)
        parameters = ScriptParameters(params)
        parameters.class_list.parameterType = "Required"
        parameters.class_list.displayName = "Classes (will be merged)"

        max_distance = arcpy.Parameter(
            displayName="Maximum connection distance (map units)",
            name="conn_distance",
            datatype="Double",
            parameterType="Required",
            direction="Input",
        )
        params.insert(2, max_distance)

        max_allowable_distance_param = arcpy.Parameter(
            displayName="Maximum Allowable Offset (feature units)",
            name="max_allowable_distance",
            datatype="Long",
            parameterType="Required",
            direction="Input",
        )
        max_allowable_distance_param.value = 150
        params.append(max_allowable_distance_param)

        out_connections_param = arcpy.Parameter(
            displayName="Output connections layer",
            name="out_connections",
            datatype="Shapefile",
            parameterType="Optional",
            direction="Output",
        )
        params.append(out_connections_param)

        return params

    def updateParameters(self, params):
        MetricsCalcTool.updateParameters(self, params)

    def _copy_fid_to_patch_id(self, selected_features):
        """ Copy FID of selected features to separate column """
        arcpy.AddField_management(selected_features, 'patchID', 'LONG')
        with UpdateCursor(selected_features, ['FID', 'patchID']) as selectedFeaturesCursor:
            for row in selectedFeaturesCursor:
                row[1] = row[0]
                selectedFeaturesCursor.updateRow(row)

    def ___prepare_input_layers(self, input_area, stat_layer, class_field, class_list):
        stat_layer_dissolved = self.create_temp_layer('stat_layer_dissolved')
        arcpy.Dissolve_management(stat_layer, stat_layer_dissolved)
        selected_features = self.create_temp_layer('selected_features')
        select_features_from_feature_class(class_field, class_list, input_area, selected_features)
        union_features = self.create_temp_layer('union_features')
        arcpy.Union_analysis([selected_features], union_features, join_attributes='ONLY_FID')
        input_layer_dissolved = self.create_temp_layer('input_layer_dissolved')
        arcpy.Dissolve_management(union_features, input_layer_dissolved, multi_part=False)
        return input_layer_dissolved, selected_features, stat_layer_dissolved

    def _generate_connections(self, params):
        prepared_input_layers = self.___prepare_input_layers(params.input_area, params.stat_layer, params.class_field,
                                                             params.class_list)
        input_layer_dissolved, selected_features, stat_layer_dissolved = prepared_input_layers
        log('Generating connections')

        input_simplified = self._simplify_input(input_layer_dissolved, params.conn_distance,
                                                params.max_allowable_distance)

        patches_near = self.create_temp_layer('patches_near')
        arcpy.GenerateNearTable_analysis(in_features=input_simplified,
                                         near_features=input_simplified,
                                         out_table=patches_near,
                                         search_radius=params.conn_distance,
                                         closest=False)

        connections_polygons = self.create_temp_layer('connections_polygons')
        arcpy.CreateFeatureclass_management(out_path=os.path.dirname(connections_polygons),
                                            out_name=os.path.splitext(os.path.basename(connections_polygons))[0],
                                            geometry_type='POLYGON',
                                            spatial_reference=arcpy.Describe(input_simplified).spatialReference)
        arcpy.AddField_management(in_table=connections_polygons, field_name='FID1', field_type='LONG')
        arcpy.AddField_management(in_table=connections_polygons, field_name='FID2', field_type='LONG')

        cnt = arcpy.GetCount_management(patches_near)[0]
        connections_cur = arcpy.da.InsertCursor(connections_polygons, ['SHAPE@', 'FID1', 'FID2'])
        current = 0

        done = set()
        arcpy.SetProgressor('step', 'Creating connections', 0, int(cnt), 1)

        for in_fid, near_fid in arcpy.da.SearchCursor(patches_near, ['IN_FID', 'NEAR_FID']):
            current += 1
            arcpy.SetProgressorLabel('Creating connections %s/%s' % (current, cnt))
            arcpy.SetProgressorPosition(current)

            fids = (in_fid, near_fid)
            if fids in done or tuple(reversed(fids)) in done:
                continue
            log('Creating connection polygon for FIDs: %s, %s' % (in_fid, near_fid))

            done.add(fids)
            where_clause = 'FID=%d OR FID=%d' % fids
            g1, g2 = [row[0] for row in arcpy.da.SearchCursor(input_simplified, ['SHAPE@'], where_clause=where_clause)]

            g1_buffer = g1.buffer(params.conn_distance)
            g2_buffer = g2.buffer(params.conn_distance)

            g1_int = g1.intersect(g2_buffer, 4)
            g2_int = g2.intersect(g1_buffer, 4)

            del g1_buffer, g2_buffer

            lines = []
            for g in [g1, g2, g1_int, g2_int]:
                g_as_line = arcpy.PolygonToLine_management(g, arcpy.Geometry(), neighbor_option='IGNORE_NEIGHBORS')
                if g_as_line:
                    lines.append(g_as_line[0])
            if len(lines) != 4:
                continue
            g1_line, g2_line, g1_line_int, g2_line_int = tuple(lines)

            g1_clipped_to_buffer = g1_line.intersect(g1_line_int, 2)
            g2_clipped_to_buffer = g2_line.intersect(g2_line_int, 2)

            lines1 = arcpy.SplitLine_management(g1_clipped_to_buffer, arcpy.Geometry())
            lines2 = arcpy.SplitLine_management(g2_clipped_to_buffer, arcpy.Geometry())

            polygons = []
            # print len(lines1), len(lines2), g1_clipped_to_buffer.length, g2_clipped_to_buffer.length
            for line1, line2 in itertools.product(lines1, lines2):
                if line1 == line2:
                    continue
                line1_p1 = line1.firstPoint
                line1_p2 = line1.lastPoint
                line2_p1 = line2.firstPoint
                line2_p2 = line2.lastPoint

                distances = [
                    line1.distanceTo(line2_p1) <= params.conn_distance,
                    line1.distanceTo(line2_p2) <= params.conn_distance,

                    line2.distanceTo(line1_p1) <= params.conn_distance,
                    line2.distanceTo(line1_p2) <= params.conn_distance,
                ]

                if sum(distances) == 4:
                    lines_union = line1.union(line2)
                    polygons.append(lines_union.convexHull())
            if polygons:
                connection = arcpy.Dissolve_management(polygons, arcpy.Geometry())[0]
                # in rare cases dissolve returns polyline
                if connection.type == 'polygon':
                    connections_cur.insertRow([connection, in_fid, near_fid])
                del connection
            del polygons, g1_line, g2_line, g1_line_int, g2_line_int, g1_clipped_to_buffer, g2_clipped_to_buffer

        connections_polygons_erased = self.create_temp_layer('connections_polygons_erased')
        arcpy.Erase_analysis(in_features=connections_polygons, erase_features=input_layer_dissolved,
                             out_feature_class=connections_polygons_erased)

        lines_buffers_cleaned = self._create_connections_buffers(connections_polygons_erased,
                                                                 input_layer_dissolved, input_simplified,
                                                                 params)

        arcpy.Append_management(inputs=lines_buffers_cleaned,
                                target=connections_polygons_erased,
                                schema_type='NO_TEST')

        connections_polygons_erased_dissolved = self.create_temp_layer('connections_polygons_erased_dissolved')
        arcpy.Dissolve_management(in_features=connections_polygons_erased,
                                  out_feature_class=connections_polygons_erased_dissolved)

        connections_polygons_erased_2 = self.create_temp_layer('connections_polygons_erased_2')
        arcpy.Erase_analysis(in_features=connections_polygons_erased_dissolved, erase_features=input_layer_dissolved,
                             out_feature_class=connections_polygons_erased_2)

        singlepart_connections = self.create_temp_layer('singlepart_connections')
        arcpy.MultipartToSinglepart_management(in_features=connections_polygons_erased_2,
                                               out_feature_class=singlepart_connections)

        near_table = self.create_temp_layer('near_table')
        arcpy.GenerateNearTable_analysis(in_features=singlepart_connections,
                                         near_features=input_layer_dissolved,
                                         out_table=near_table,
                                         search_radius=0,
                                         closest=False)
        near_table_count = self.create_temp_layer('near_table_count')
        arcpy.Frequency_analysis(in_table=near_table,
                                 out_table=near_table_count,
                                 frequency_fields='IN_FID')
        to_remove = set()
        with arcpy.da.SearchCursor(near_table_count,
                                   field_names=['IN_FID'],
                                   where_clause='FREQUENCY=1') as count_cursor:
            for row in count_cursor:
                to_remove.add(row[0])
        log_debug('Loose connections to remove: %s' % to_remove)
        connections_cleaned = self.create_temp_layer('connections_cleaned')
        arcpy.CreateFeatureclass_management(out_path=os.path.dirname(connections_cleaned),
                                            out_name=os.path.splitext(os.path.basename(connections_cleaned))[0],
                                            geometry_type='POLYGON',
                                            spatial_reference=arcpy.Describe(input_simplified).spatialReference)
        connections_cleaned_cur = arcpy.da.InsertCursor(connections_cleaned, ['OID@', 'SHAPE@'])
        search_cursor = arcpy.da.SearchCursor(singlepart_connections,
                                              field_names=['OID@', 'SHAPE@'])
        for conn_row in search_cursor:
            fid = conn_row[0]
            if fid not in to_remove:
                connections_cleaned_cur.insertRow(conn_row)

        connection_final_dissolved = self.create_temp_layer('connection_final_dissolved')
        arcpy.Dissolve_management(connections_cleaned, connection_final_dissolved)

        connections = self.create_temp_layer('connections')
        arcpy.Clip_analysis(in_features=connection_final_dissolved,
                            clip_features=stat_layer_dissolved,
                            out_feature_class=connections)

        log('Created connections')
        if params.out_connections:
            arcpy.CopyFeatures_management(connections, params.out_connections)

        return connections

    def _create_connections_buffers(self, connections_polygons_erased, input_layer_dissolved, input_simplified, params):
        connections_polygons_lines = self.create_temp_layer('connections_polygons_lines')
        arcpy.PolygonToLine_management(in_features=connections_polygons_erased,
                                       out_feature_class=connections_polygons_lines, neighbor_option=False)
        simplified_lines = self.create_temp_layer('simplified_lines')
        arcpy.PolygonToLine_management(in_features=input_simplified,
                                       out_feature_class=simplified_lines, neighbor_option=False)
        connections_border_lines_multi = self.create_temp_layer('connections_border_lines_multi')
        arcpy.Intersect_analysis(in_features=[connections_polygons_lines, simplified_lines],
                                 out_feature_class=connections_border_lines_multi)
        connections_border_lines = self.create_temp_layer('connections_border_lines')
        arcpy.MultipartToSinglepart_management(in_features=connections_border_lines_multi,
                                               out_feature_class=connections_border_lines)
        connections_border_lines_buffer_1 = self.create_temp_layer('connections_border_lines_buffer_1')
        connections_border_lines_buffer_2 = self.create_temp_layer('connections_border_lines_buffer_2')
        connections_border_lines_buffer = self.create_temp_layer('connections_border_lines_buffer')
        arcpy.Buffer_analysis(in_features=connections_border_lines,
                              out_feature_class=connections_border_lines_buffer_1,
                              buffer_distance_or_field=params.max_allowable_distance,
                              line_end_type='FLAT',
                              line_side='LEFT')
        arcpy.Buffer_analysis(in_features=connections_border_lines,
                              out_feature_class=connections_border_lines_buffer_2,
                              buffer_distance_or_field=params.max_allowable_distance,
                              line_end_type='FLAT',
                              line_side='RIGHT')
        arcpy.Append_management(inputs=connections_border_lines_buffer_1,
                                target=connections_border_lines_buffer_2,
                                schema_type='NO_TEST')
        arcpy.Sort_management(in_dataset=connections_border_lines_buffer_2,
                              out_dataset=connections_border_lines_buffer,
                              sort_field=[['ORIG_FID', 'ASCENDING']])
        arcpy.AddField_management(in_table=connections_border_lines_buffer, field_name='BUFF_ID', field_type='LONG')
        cursor = arcpy.da.UpdateCursor(connections_border_lines_buffer, field_names=['OID@', 'BUFF_ID'])
        for row in cursor:
            row[1] = row[0]
            cursor.updateRow(row)
        buffer_input_tabulate_intersect = self.create_temp_layer('buffer_tabulate_intersect')
        arcpy.TabulateIntersection_analysis(
            in_zone_features=connections_border_lines_buffer,
            zone_fields=['BUFF_ID'],
            in_class_features=input_layer_dissolved,
            out_table=buffer_input_tabulate_intersect
        )
        arcpy.JoinField_management(in_data=connections_border_lines_buffer,
                                   in_field='BUFF_ID',
                                   join_table=buffer_input_tabulate_intersect,
                                   join_field='BUFF_ID')

        lines_buffers_cleaned = self.create_temp_layer('lines_buffers_cleaned')
        arcpy.CreateFeatureclass_management(out_path=os.path.dirname(lines_buffers_cleaned),
                                            out_name=os.path.splitext(os.path.basename(lines_buffers_cleaned))[0],
                                            geometry_type='POLYGON',
                                            spatial_reference=arcpy.Describe(input_simplified).spatialReference)

        search_cursor = arcpy.da.SearchCursor(connections_border_lines_buffer,
                                              field_names=['ORIG_FID', 'PERCENTAGE', 'SHAPE@'])
        with arcpy.da.InsertCursor(lines_buffers_cleaned, field_names=['SHAPE@']) as insert_cursor:
            for key, rrs in itertools.groupby(search_cursor, key=lambda r: r[0]):
                overlapping_row = max(rrs, key=lambda r: r[1])
                insert_cursor.insertRow([overlapping_row[2]])

        return lines_buffers_cleaned

    class ConnectanceMetricsParameters(object):

        def __init__(self, tool, parameters):
            super(ConnectanceMetricsTool.ConnectanceMetricsParameters, self).__init__()

            parameters_map = dict([(param.name, param) for param in parameters])

            self.input_area = parameters_map['in_area'].valueAsText
            self.stat_layer = parameters_map['stat_layer'].valueAsText
            self.class_field_name = parameters_map['class_field'].valueAsText
            self.class_field = get_field(self.input_area, self.class_field_name)
            self.conn_distance = parameters_map['conn_distance'].value
            self.out_connections = parameters_map['out_connections'].valueAsText
            self.max_allowable_distance = int(parameters_map['max_allowable_distance'].value)

            class_list = getParameterValues(parameters_map['class_list'])
            if not class_list or len(class_list) == 0:
                class_list = tool.get_all_classes(self.input_area, self.class_field_name)
            self.class_list = map(str, class_list)

            selected_features = tool.create_temp_layer('selected_features')
            select_features_from_feature_class(self.class_field, class_list, self.input_area, selected_features)
            self.selected_features = selected_features

    def execute(self, parameters, messages):
        setup_debug_mode()

        params = self.ConnectanceMetricsParameters(self, parameters)

        self.prepare_stat_layer(params.stat_layer,
                                EnsureUnitIDFieldExists(),
                                EnsureZoneAreaFieldExists(),
                                EnsureFieldExists(field_name=self.NUMBER_OF_PATCHES_FIELD_NAME, field_type='LONG',
                                                  default_value=0),
                                EnsureFieldExists(field_name=self.CONNECTION_AREA_FIELD_NAME, field_type='DOUBLE',
                                                  default_value=0),
                                EnsureFieldExists(field_name=self.CONNECTION_AREA_PERCENTAGE_FIELD_NAME,
                                                  field_type='DOUBLE', default_value=0),
                                EnsureFieldExists(field_name=self.PATCH_AREA_FIELD_NAME, field_type='DOUBLE',
                                                  default_value=0),
                                EnsureFieldExists(field_name=self.PATCH_AREA_PERCENT_FIELD_NAME, field_type='DOUBLE',
                                                  default_value=0), )

        try:
            msg = "Analyzing"
            log(msg)
            arcpy.SetProgressor("step", msg, 0, int(arcpy.GetCount_management(params.stat_layer).getOutput(0)) * 2, 1)

            layer_to_analyse = self._prepare_layer_to_analyse(params)
            connections = self._generate_connections(params)

            self._analyse_layer(params, layer_to_analyse, connections)

        except Exception, e:
            handleException(e)
        finally:
            arcpy.ResetProgressor()
            self.on_exit()

    def _prepare_layer_to_analyse(self, params):
        input_layer_dissolved = self.create_temp_layer('input_layer_dissolved')
        arcpy.Dissolve_management(in_features=params.selected_features,
                                  out_feature_class=input_layer_dissolved,
                                  multi_part='SINGLE_PART')

        buffer_around_input = self.create_temp_layer('buffer_around_input')
        arcpy.Buffer_analysis(in_features=input_layer_dissolved,
                              out_feature_class=buffer_around_input,
                              buffer_distance_or_field=params.conn_distance,
                              line_side='OUTSIDE_ONLY',
                              line_end_type='ROUND',
                              dissolve_option='NONE')

        layer_to_analyse = self.create_temp_layer('layer_to_analyse')
        arcpy.Intersect_analysis(in_features=[buffer_around_input, params.selected_features],
                                 out_feature_class=layer_to_analyse,
                                 join_attributes='NO_FID')

        self._copy_fid_to_patch_id(layer_to_analyse)
        return layer_to_analyse

    @staticmethod
    def _get_patches_number_and_area(stat_input_intersection, unit_id):
        distinct_patches = set()
        patches_area = 0
        with SearchCursor(stat_input_intersection,
                          field_names=['SHAPE@AREA', 'patchID', UNIT_ID_FIELD_NAME],
                          where_clause='%s=%d' % (UNIT_ID_FIELD_NAME, unit_id)) as sc1:
            for r1 in sc1:
                patch_area = r1[0]
                patches_area += patch_area
                distinct_patches.add(r1[1])
        number_of_patches = len(distinct_patches)
        return number_of_patches, patches_area

    @staticmethod
    def _get_connections_area(stat_connections_intersection, unit_id):
        with SearchCursor(stat_connections_intersection, field_names=['SHAPE@AREA', 'unitID'],
                          where_clause='%s=%d' % (UNIT_ID_FIELD_NAME, unit_id)) as sc2:
            return sum([r2[0] for r2 in sc2])

    def _analyse_layer(self, params, layer_to_analyse, connections):
        stat_input_intersection = self.create_temp_layer('stat_input_intersection')
        stat_connections_intersection = self.create_temp_layer('stat_connections_intersection')
        arcpy.Intersect_analysis(in_features=[params.stat_layer, layer_to_analyse],
                                 out_feature_class=stat_input_intersection)
        arcpy.Intersect_analysis(in_features=[params.stat_layer, connections],
                                 out_feature_class=stat_connections_intersection)

        with UpdateCursor(params.stat_layer, [UNIT_ID_FIELD_NAME,  # 0 -hex ID
                                              ZONE_AREA_FIELD_NAME,  # 1 - hex area
                                              self.NUMBER_OF_PATCHES_FIELD_NAME,  # 2
                                              self.CONNECTION_AREA_FIELD_NAME,  # 3
                                              self.CONNECTION_AREA_PERCENTAGE_FIELD_NAME,  # 4
                                              self.PATCH_AREA_FIELD_NAME,  # 5
                                              self.PATCH_AREA_PERCENT_FIELD_NAME  # 6
                                              ]) as stat_layer_cursor:
            for stat_layer_row in stat_layer_cursor:
                unit_id = stat_layer_row[0]
                zone_area = stat_layer_row[1]

                number_of_patches, patches_area = self._get_patches_number_and_area(stat_input_intersection, unit_id)
                connections_area = self._get_connections_area(stat_connections_intersection, unit_id)

                stat_layer_row[2] = number_of_patches
                stat_layer_row[3] = connections_area
                stat_layer_row[4] = round(connections_area / zone_area * 100, 3)
                stat_layer_row[5] = patches_area
                stat_layer_row[6] = round(patches_area / zone_area * 100, 3)

                stat_layer_cursor.updateRow(stat_layer_row)

    def _simplify_input(self, input_layer_dissolved, conn_distance, max_allowable_distance):
        input_simplified = self.create_temp_layer('input_simplified')
        arcpy.SimplifyPolygon_cartography(in_features=input_layer_dissolved,
                                          out_feature_class=input_simplified,
                                          algorithm='POINT_REMOVE',
                                          tolerance=max_allowable_distance,
                                          collapsed_point_option=False)
        arcpy.Densify_edit(in_features=input_simplified, densification_method='DISTANCE', distance=conn_distance / 5)
        return input_simplified


class CreateHexagons(object):
    def __init__(self):
        # see http://resources.arcgis.com/en/help/main/10.1/index.html#//001500000024000000
        self.label = "Create hexagons"
        self.description = '''Creates a new hexagon layer based on a user defined hexagon height.<br/> Hexagons may be also centered to the specific point (defined by centroid of the given layer) to represent the relations between the distance from the center and the observed values.'''
        self.canRunInBackground = True
        # self.category = 'Creating'

    def getParameterInfo(self):
        # Define parameter definitions

        inputArea = arcpy.Parameter(
            displayName="Input layer",
            name="in_area",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        useExtent = arcpy.Parameter(
            displayName="Hexagon layer extent = display extent",
            name="use_extent",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
        useExtent.value = False

        clipToInput = arcpy.Parameter(
            displayName="Clip hexagon layer to input area",
            name="clip_to_input",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
        clipToInput.value = True

        hexHeight = arcpy.Parameter(
            displayName="Hexagon height",
            name="hex_height",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        hexHeight.value = 1000

        outFeatureClass = arcpy.Parameter(
            displayName="Output hexagon Layer",
            name="output_layer",
            datatype="DEShapefile",
            parameterType="Required",
            direction="Output")

        centerHexagons = arcpy.Parameter(
            displayName="Center hexagons",
            name="center_hexagons",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input")

        centerFeatureLayer = arcpy.Parameter(
            displayName="Feature layer to center",
            name="center_fc",
            datatype="Feature Layer",
            parameterType="Optional",
            direction="Input")

        params = [inputArea, useExtent, clipToInput, hexHeight, centerHexagons, centerFeatureLayer, outFeatureClass]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        input_parameters = ScriptParameters(parameters)
        if input_parameters.in_area.altered:
            # input_parameters.center_fc.value = input_parameters.in_area.value
            pass
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        setup_debug_mode()
        input_parameters = ScriptParameters(parameters)
        width = input_parameters.hex_height.value
        hexOut = input_parameters.output_layer.valueAsText
        inputArea = input_parameters.in_area.valueAsText
        boolExtent = input_parameters.use_extent.value
        clipToInput = input_parameters.clip_to_input.value
        centerHexagons = input_parameters.center_hexagons
        centerLayer = input_parameters.center_fc.valueAsText
        if centerHexagons and not centerLayer:
            centerLayer = inputArea

        scratchworkspace = get_scratchworkspace()
        log("create hexagon layer....")
        Fishnet_1 = scratchworkspace + "\\Fishnet1"
        Fishnet_2 = scratchworkspace + "\\Fishnet2"
        Fishnet_Label_1 = scratchworkspace + "\\Fishnet1_Label"
        Fishnet_Label_2 = scratchworkspace + "\\Fishnet2_Label"
        Appended_Points_Name = "hex_points"
        Appended_Points = scratchworkspace + "\\" + Appended_Points_Name

        delete_if_exists(Fishnet_1, Fishnet_2, Appended_Points)

        # Process: Calculate Value (width)...
        height = float(width) * math.sqrt(3)

        # Invert the height and width so that the flat side of the hexagon is on the bottom and top
        tempWidth = width
        width = height
        height = tempWidth

        log("height: " + str(height))
        log("width: " + str(width))

        # Process: Create Extent Information...
        ll = self.calculateOrigin(inputArea, boolExtent)
        Origin = self.updateOrigin(ll, width, height, -2.0)
        ur = self.calculateUR(inputArea, boolExtent)

        Opposite_Corner = self.updateOrigin(ur, width, height, 2.0)
        log("origin: " + Origin)
        log("opposite corner: " + Opposite_Corner)

        # Process: Calculate Value (Origin)...
        newOrigin = self.updateOrigin(Origin, width, height, 0.5)
        log("new origin: " + newOrigin)

        # Process: Calculate Value (Opposite Corner)...
        newOpposite_Corner = self.updateOrigin(Opposite_Corner, width, height, 0.5)
        log("newOpposite_Corner: " + newOpposite_Corner)

        # Process: Calculate Value (Y Axis 1)...
        Y_Axis_Coordinates1 = self.getYAxisCoords(Origin, Opposite_Corner)
        log("Y_Axis_Coordinates1: " + Y_Axis_Coordinates1)

        # Process: Create Fishnet...
        arcpy.CreateFishnet_management(Fishnet_1, Origin, Y_Axis_Coordinates1, width, height, "0", "0", Opposite_Corner,
                                       "LABELS", "")
        log("created fishnet 1...")
        arcpy.Delete_management(Fishnet_1)

        # Process: Calculate Value (Y Axis 2)...
        YAxis_Coordinates2 = self.getYAxisCoords(newOrigin, newOpposite_Corner)
        log("YAxis_Coordinates2: " + YAxis_Coordinates2)

        # Process: Calculate Value (Number of Columns)...
        Number_of_Columns = self.getCols(Origin, width, Opposite_Corner)
        log("Number_of_Columns: " + str(Number_of_Columns))

        # Process: Create Fishnet (2)...
        arcpy.CreateFishnet_management(Fishnet_2, newOrigin, YAxis_Coordinates2, width, height, "0", "0",
                                       newOpposite_Corner, "LABELS", "")
        log("created fishnet 2...")
        arcpy.Delete_management(Fishnet_2)

        # Process: Create Feature Class...
        arcpy.CreateFeatureclass_management(scratchworkspace, Appended_Points_Name, "POINT", "#", "#", "#",
                                            arcpy.Describe(inputArea).SpatialReference)
        log("created template fc...")

        # Process: Append...
        arcpy.Append_management(Fishnet_Label_1 + ";" + Fishnet_Label_2, Appended_Points, "NO_TEST", "", "")
        log("appended fishnets...")
        arcpy.Delete_management(Fishnet_Label_1)
        arcpy.Delete_management(Fishnet_Label_2)

        with createTempLayer('hexBeforeClip') as hexBeforeClip:

            # Process: Create Thiessen Polygons...
            # Limit Hexagons roughly to the real input data extent:
            arcpy.MakeFeatureLayer_management(Appended_Points, "in_memory\\hexPoints")
            arcpy.SelectLayerByLocation_management("in_memory\\hexPoints", "WITHIN_A_DISTANCE", inputArea, width * 2,
                                                   "NEW_SELECTION")
            arcpy.CreateThiessenPolygons_analysis("in_memory\\hexPoints", hexBeforeClip, "ONLY_FID")
            log("created thiessen polygons...")
            arcpy.Delete_management(Appended_Points)

            if centerHexagons:
                self.centerHexagons(centerLayer, hexBeforeClip)

            # Process: keep Hexagons only
            # if boolExtent == "false":
            if not clipToInput:
                log("keep only hexagons...")
                self.deleteNotHexagons(hexBeforeClip, inputArea, height, boolExtent)
                arcpy.CopyFeatures_management(hexBeforeClip, hexOut)
            else:
                log("clipping to input layer area...")
                arcpy.Clip_analysis(hexBeforeClip, inputArea, out_feature_class=hexOut)

            arcpy.AddField_management(hexOut, UNIT_ID_FIELD_NAME, "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED",
                                      "")
            log("added field unitID...")

            # Process: Calculate Hexagonal Polygon ID's...
            numberHexagons = self.calculateHexPolyID(hexOut)
            log("number of Units = " + str(numberHexagons))

            # Process: Add Spatial Index...
            gdb = os.path.dirname(hexOut)
            if gdb.find("mdb") <> -1:
                log("personal")
            else:
                arcpy.AddSpatialIndex_management(hexOut)
                log("added Spatial Index...")

            log("Units prepared.....")

    def calculateHexPolyID(self, hexOut):
        fields = (UNIT_ID_FIELD_NAME,)
        with UpdateCursor(hexOut, fields) as cur:
            ID = 1
            for row in cur:
                row[0] = ID
                cur.updateRow(row)
                ID += 1
        return ID - 1

    def calculateOrigin(self, dataset, boolExtent):
        if boolExtent == "true":
            mxd = arcpy.mapping.MapDocument("CURRENT")
            df = mxd.activeDataFrame
            ext = df.extent
            return str(ext.XMin) + " " + str(ext.YMin)
        else:
            ext = arcpy.Describe(dataset).extent
            # coords = ext.split(" ")
            return str(ext.XMin) + " " + str(ext.YMin)

    def calculateUR(self, dataset, boolExtent):
        if boolExtent == "true":
            mxd = arcpy.mapping.MapDocument("CURRENT")
            df = mxd.activeDataFrame
            ext = df.extent
            return str(ext.XMax) + " " + str(ext.YMax)
        else:
            ext = arcpy.Describe(dataset).extent
            # coords = ext.split(" ")
            return str(ext.XMax) + " " + str(ext.YMax)

    def calculateHeight(self, width):
        return float(width) * math.sqrt(3)

    def updateOrigin(self, origin, width, height, factor):
        coords = origin.split(" ")
        xcoord = float(coords[0])
        ycoord = float(coords[1])
        return str(xcoord + float(width) * factor) + " " + str(ycoord + float(height) * factor)

    def getYAxisCoords(self, origin, opposite):
        origin_coords = origin.split(" ")
        xcoord_origin = float(origin_coords[0])
        corner_coords = opposite.split(" ")
        ycoord_opposite = float(corner_coords[1])
        return str(xcoord_origin) + " " + str(ycoord_opposite)

    def getCols(self, origin, width, opposite):
        coords = origin.split(" ")
        x_origin = float(coords[0])
        coords = opposite.split(" ")
        x_opposite = float(coords[0])
        return int((x_opposite - x_origin) / int(width))

    def deleteNotHexagons(self, input1, input2, height, boolExtent):
        geo = arcpy.Geometry()
        geometryList = arcpy.CopyFeatures_management(input2, geo)
        fuzzyArea = 2 * ((float(height) / 2) ** 2) * (3 ** 0.5)
        log("fuzzyArea: " + str(fuzzyArea))
        # unionpoly = arcpy.Polygon()
        i = 0
        for poly in geometryList:
            if i == 0:
                unionpoly = poly
            else:
                unionpoly = unionpoly.union(poly)
            i = 1

        x = 0
        with UpdateCursor(input1, ["SHAPE@"]) as cur:
            for row in cur:
                feat = row[0]
                if boolExtent == "false" and feat.disjoint(unionpoly):
                    cur.deleteRow()
                    x += 1
                elif feat.area > (fuzzyArea + 10) or feat.area < (fuzzyArea - 10):
                    cur.deleteRow()
                    x += 1
        log("deleted hexagons: " + str(x))

    def centerHexagons(self, centerLayer, hexesLayer):
        # move hexagons to center of input layer
        geometryList = arcpy.MinimumBoundingGeometry_management(in_features=centerLayer,
                                                                out_feature_class=arcpy.Geometry(),
                                                                geometry_type='CIRCLE', group_option='ALL',
                                                                mbg_fields_option='MBG_FIELDS')
        centroidOfInput = geometryList[0].centroid
        centroidOfHexes = centroidOfInput

        arcpy.AddField_management(hexesLayer, 'center_dst', "LONG")

        # search for center hexagon
        # TODO - should be a better way of finding center hexagon
        centerHexId = None
        with UpdateCursor(hexesLayer, ['SHAPE@', 'OID@', 'center_dst']) as sc:
            for row in sc:
                g = row[0]
                if g.contains(centroidOfInput):
                    centroidOfHexes = g.centroid
                    centerHexId = row[1]
                    row[2] = 0
                else:
                    row[2] = -1
                sc.updateRow(row)

        dx = centroidOfHexes.X - centroidOfInput.X
        dy = centroidOfHexes.Y - centroidOfInput.Y
        with UpdateCursor(hexesLayer, ['SHAPE@']) as sc:
            for row in sc:
                array = arcpy.Array()
                # Step through each part of the feature
                for part in row[0]:
                    # Step through each vertex in the feature
                    for pnt in part:
                        if pnt:
                            x = pnt.X - dx
                            y = pnt.Y - dy
                            np = arcpy.Point(x, y)
                            np.ID = pnt.ID
                            array.add(np)

                polygon = arcpy.Polygon(array)
                row[0] = polygon
                sc.updateRow(row)
        log('centerHexId: ' + str(centerHexId))

        with createTempLayer('hexSelection') as tmpSel:
            arcpy.MakeFeatureLayer_management(hexesLayer, tmpSel)

            with createTempLayer('previousHex') as previousHex:
                dst = 0
                arcpy.SelectLayerByAttribute_management(tmpSel, selection_type='NEW_SELECTION',
                                                        where_clause='OID=%d' % centerHexId)
                arcpy.CopyFeatures_management(tmpSel, previousHex)

                oids = [centerHexId]
                while oids:
                    dst += 1
                    log('Searching for hexagons within distance of %d units from center' % dst)
                    arcpy.SelectLayerByLocation_management(tmpSel, overlap_type='BOUNDARY_TOUCHES',
                                                           select_features=previousHex, selection_type='NEW_SELECTION')
                    arcpy.SelectLayerByAttribute_management(tmpSel, selection_type='SUBSET_SELECTION',
                                                            where_clause='center_dst=-1')
                    delete_if_exists(previousHex)
                    arcpy.CopyFeatures_management(tmpSel, previousHex)
                    oids = [str(row[0]) for row in SearchCursor(tmpSel, ['OID@'])]
                    # log('found oids: ' + str(oids))
                    if not oids:
                        # no more hexagons
                        break
                    with UpdateCursor(hexesLayer, field_names=['OID@', 'center_dst', '*'],
                                      where_clause='OID IN (%s)' % (','.join(oids))) as uc:
                        for row in uc:
                            row[1] = dst
                            uc.updateRow(row)


                            # arcpy.Delete_management(hexesLayer)
                            # arcpy.CopyFeatures_management(tmpSel, hexesLayer)


class CreatePie(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        # see http://resources.arcgis.com/en/help/main/10.1/index.html#//001500000024000000
        self.label = "Create pie layer"
        self.description = """Creates layer dividing the input area (defined by the to be analysed layer) into a defined number of sections (similar to pie charts, but with equal arc length).<br/>
        The sections are geographically oriented, that means that the first sector is always directed to the North, and e. g. when 4 sections would set in the properties, the whole area will be divided to the 90 degree sections directed to the four sides of the world."""
        self.canRunInBackground = True
        # self.category = 'Creating'

    def getParameterInfo(self):
        # Define parameter definitions

        # Input layer - a layer defining the extent of the pie layer (usually the layer for which the metrics will be calculated)
        inputArea = arcpy.Parameter(
            displayName="Input layer",
            name="in_area",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        # Number of sections - the number of pie sections to be calculated. Pie section will be geograpically oriented, the first is directed to the North. Usually  numbers divisible by 4 are recommended
        sections_number = arcpy.Parameter(
            displayName="Number of sections",
            name="sections_number",
            datatype="Long",
            parameterType="Required",
            direction="Input")
        sections_number.value = 8

        # Output pie layer - the name for the output pie layer
        outFeatureClass = arcpy.Parameter(
            displayName="Output pie layer",
            name="output_layer",
            datatype="DEShapefile",
            parameterType="Required",
            direction="Output")

        params = [inputArea, sections_number, outFeatureClass]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        parameters[1].clearMessage()
        if int(parameters[1].value) < 2:
            parameters[1].setErrorMessage('Number of sections must be value greater than 1')
        return

    def execute(self, parameters, messages):
        setup_debug_mode()
        import cmath

        input_parameters = ScriptParameters(parameters)
        inputArea = input_parameters.in_area.valueAsText
        n = int(input_parameters.sections_number.valueAsText)
        out = input_parameters.output_layer.valueAsText

        bound = 'in_memory\\bound_circle%d' % random.randint(0, 1000000)
        points = 'bound_point%d' % random.randint(0, 1000000)
        thies = 'in_memory\\thiessen%d' % random.randint(0, 1000000)
        try:
            arcpy.CreateFeatureclass_management(get_scratchworkspace(), points, "POINT", "#", "#", "#",
                                                arcpy.Describe(inputArea).SpatialReference)
            points = 'in_memory\\' + points
            arcpy.MinimumBoundingGeometry_management(inputArea, bound, "CIRCLE", "ALL", mbg_fields_option="MBG_FIELDS")
            arcpy.AddField_management(points, UNIT_ID_FIELD_NAME, "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED",
                                      "")
            d = 0
            centroid = None
            with SearchCursor(bound, ('MBG_Diameter', 'SHAPE@TRUECENTROID')) as c:
                row = c.next()
                d = row[0]
                centroid = row[1]

            phiInRadians = 2 * math.pi / n
            r = d / 2.
            initialAngle = math.pi / 2

            with InsertCursor(points, ('SHAPE@', UNIT_ID_FIELD_NAME)) as cur:
                for i in xrange(0, n):
                    rect = cmath.rect(r, phiInRadians * i + initialAngle)
                    rect = (rect.real + centroid[0], rect.imag + centroid[1])

                    point = arcpy.Point()
                    point.X = rect[0]
                    point.Y = rect[1]
                    cur.insertRow((point, i + 1))

            arcpy.CreateThiessenPolygons_analysis(points, thies, "ALL")
            arcpy.Clip_analysis(thies, inputArea, out_feature_class=out)

        except Exception, e:
            handleException(e)
        finally:
            delete_if_exists(points, bound, thies)
