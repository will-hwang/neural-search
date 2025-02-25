/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.optimization;

import lombok.extern.log4j.Log4j2;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.env.Environment;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.processor.InferenceProcessor;
import org.opensearch.neuralsearch.processor.util.ProcessorUtils;
import org.opensearch.neuralsearch.util.ProcessorDocumentUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The abstract class for selective text processing use cases. On update operation, the selective inference processor will attempt to
 * skip inference calls by copying over existing embeddings for the same text
 */

@Log4j2
public abstract class SelectiveInferenceProcessor extends InferenceProcessor {
    protected Map<String, String> reversedFieldMap;

    public SelectiveInferenceProcessor(
        String tag,
        String description,
        int batchSize,
        String modelId,
        String type,
        String listTypeNestedMapKey,
        Map<String, Object> fieldMap,
        MLCommonsClientAccessor clientAccessor,
        Environment environment,
        ClusterService clusterService
    ) {
        super(tag, description, batchSize, type, listTypeNestedMapKey, modelId, fieldMap, clientAccessor, environment, clusterService);
        this.reversedFieldMap = ProcessorDocumentUtils.flattenAndFlip(fieldMap);
    }

    public abstract Object processValue(
        String currentPath,
        Object processValue,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap,
        int index
    );

    public abstract List<Object> processValues(
        List<Object> processList,
        Optional<Object> sourceList,
        Optional<Object> existingList,
        Optional<Object> embeddingList,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap,
        String fullEmbeddingKey
    );

    public Map<String, Object> filterProcessMap(
        Map<String, Object> existingSourceAndMetadataMap,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> processMap
    ) {
        return filterProcessMap(existingSourceAndMetadataMap, sourceAndMetadataMap, processMap, "", 0);
    }

    /**
     * Filters and processes a nested map structure by comparing values between existing and new metadata maps.
     *
     * @param existingSourceAndMetadataMap SourceAndMetadataMap of existing Document
     * @param sourceAndMetadataMap SourceAndMetadataMap of ingestDocument Document
     * @param processMap The current processMap
     * @param prevPath The dot-notation path of the parent elements
     * @param prevLevel The current nesting level in the hierarchy
     * @return A filtered map containing only the elements that differ between the existing and new metadata maps
     *
     */
    protected Map<String, Object> filterProcessMap(
        Map<String, Object> existingSourceAndMetadataMap,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> processMap,
        String prevPath,
        int prevLevel
    ) {
        Map<String, Object> filteredProcessMap = new HashMap<>();

        for (Map.Entry<String, Object> entry : processMap.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            String currentPath = prevPath.isEmpty() ? key : prevPath + "." + key;
            int currLevel = prevLevel + 1;
            if (value instanceof Map) {
                Map<String, Object> filteredInnerMap = filterProcessMap(
                    existingSourceAndMetadataMap,
                    sourceAndMetadataMap,
                    (Map) value,
                    currentPath,
                    currLevel
                );
                filteredProcessMap.put(key, filteredInnerMap.isEmpty() ? null : filteredInnerMap);
            } else if (value instanceof List) {
                List<Object> processedList = processListValue(
                    currentPath,
                    (List<Object>) value,
                    sourceAndMetadataMap,
                    existingSourceAndMetadataMap
                );
                if (!processedList.isEmpty()) {
                    filteredProcessMap.put(key, processedList);
                }
            } else {
                Object processedValue = processValue(currentPath, value, sourceAndMetadataMap, existingSourceAndMetadataMap, -1);
                filteredProcessMap.put(key, processedValue);
            }
        }
        return filteredProcessMap;
    }

    /**
     * Processes a list of values by comparing them against source and existing metadata.
     *
     * @param currentPath The current path in dot notation for the list being processed
     * @param processList The list of values to process
     * @param sourceAndMetadataMap SourceAndMetadataMap of ingestDocument Document
     * @param existingSourceAndMetadataMap SourceAndMetadataMap of existing Document
     * @return A processed list containing non-filtered elements
     */
    protected List<Object> processListValue(
        String currentPath,
        List<Object> processList,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap
    ) {
        String textKey = reversedFieldMap.get(currentPath);
        if (textKey == null) {
            return processList;
        }

        Optional<Object> sourceList = ProcessorUtils.getValueFromSource(sourceAndMetadataMap, textKey);
        Optional<Object> existingList = ProcessorUtils.getValueFromSource(existingSourceAndMetadataMap, textKey);
        Optional<Object> embeddingList = ProcessorUtils.getValueFromSource(existingSourceAndMetadataMap, currentPath);
        if (sourceList.isPresent() && sourceList.get() instanceof List) {
            return processValues(
                processList,
                sourceList,
                existingList,
                embeddingList,
                sourceAndMetadataMap,
                existingSourceAndMetadataMap,
                currentPath
            );
        } else {
            return processMapValuesInList(processList, currentPath, sourceAndMetadataMap, existingSourceAndMetadataMap);

        }
    }

    /**
     * Processes a list containing map values by iterating through each item and processing it individually.
     *
     * @param processList The list of Map items to process
     * @param currentPath The current path in dot notation
     * @param sourceAndMetadataMap SourceAndMetadataMap of ingestDocument Document
     * @param existingSourceAndMetadataMap SourceAndMetadataMap of existing Document
     * @return A processed list containing non-filtered elements
     */
    private List<Object> processMapValuesInList(
        List<Object> processList,
        String currentPath,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap
    ) {
        List<Object> filteredList = new ArrayList<>();
        for (int i = 0; i < processList.size(); i++) {
            Object processedItem = processValue(currentPath, processList.get(i), sourceAndMetadataMap, existingSourceAndMetadataMap, i);
            filteredList.add(processedItem);
        }
        return filteredList;
    }
}
