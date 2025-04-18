/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.optimization;

import lombok.extern.log4j.Log4j2;
import org.opensearch.neuralsearch.processor.util.ProcessorUtils;
import org.opensearch.neuralsearch.util.ProcessorDocumentUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Abstract class for selective text processing and embedding optimization.
 * The InferenceFilter class is designed optimize inference calls by selectively processing text data.
 * It reuses existing embeddings when the text content remains unchanged, reducing redundant inference calls and
 * improving performance. This is achieved by comparing the text in new and existing documents, copying embeddings
 * when eligible.
 * This class is intended to be extended for different processing use cases. It provides a recursive filtering
 * mechanism that navigates through nested map structures, comparing values, and determining if embeddings can be reused.
 */
@Log4j2
public abstract class InferenceFilter {
    /**
     * Stores the reverse mapping of field names to support efficient lookups for embedding keys.
     * This is generated by flattening and flipping the provided field map.
     */
    protected Map<String, String> reversedFieldMap;

    /**
     * Constructs an InferenceFilter instance and initializes the reversed field map.
     */
    public InferenceFilter(Map<String, Object> fieldMap) {
        this.reversedFieldMap = ProcessorDocumentUtils.flattenAndFlip(fieldMap);
    }

    /**
     * Abstract method to filter individual values based on the existing and new metadata maps.
     * Implementations should provide logic to compare values and determine if embeddings can be reused.
     *
     * @param embeddingKey The dot-notation path for the embedding field
     * @param processValue The value to be checked for potential embedding reuse
     * @param sourceAndMetadataMap The metadata map of the new document
     * @param existingSourceAndMetadataMap The metadata map of the existing document
     * @param index The index to be inserted into if list exists in the new document
     * @return The processed value or null if embeddings are reused
     */
    public abstract Object filterInferenceValue(
        String embeddingKey,
        Object processValue,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap,
        int index
    );

    /**
     * Abstract helper method to filter individual objects based on the existing and new metadata maps.
     * Implementations should provide logic to compare objects and determine if embeddings can be reused.
     *
     * @param embeddingKey The dot-notation path for the embedding field
     * @param processValue The value to be checked for potential embedding reuse
     * @param existingValue The existing value for comparison
     * @param embeddingValue The existing embedding value
     * @param sourceAndMetadataMap The metadata map of the new document
     * @param index The index to be inserted into if list exists in the new document
     * @return The processed value or null if embeddings are reused
     */

    public abstract Object copyEmbeddingForSingleObject(
        String embeddingKey,
        Object processValue,
        Object existingValue,
        Object embeddingValue,
        Map<String, Object> sourceAndMetadataMap,
        int index
    );

    /**
     * Abstract method to filter and compare lists of objects.
     * If all objects in the list are identical between the new and existing metadata maps, embeddings are copied,
     * and null is returned to indicate no further processing is required.
     *
     * @param embeddingKey The dot-notation path for the embedding field
     * @param processList The list of values to be checked for potential embedding reuse
     * @param existingList The list of existing values for comparison
     * @param embeddingList The list of existing embeddings
     * @param sourceAndMetadataMap The metadata map of the new document.
     * @return A processed list or an empty list if embeddings are reused.
     */

    public abstract List<Object> copyEmbeddingForListObject(
        String embeddingKey,
        List<Object> processList,
        List<Object> existingList,
        List<Object> embeddingList,
        Map<String, Object> sourceAndMetadataMap
    );

    /**
     * This method navigates through the nested structure, checking each key-value pair recursively. It supports:
     * Map values: Processed recursively using this method.
     * List values: Processed using filterListValue.
     * String values: Directly compared using filterInferenceValue.
     *
     * @param existingSourceAndMetadataMap The metadata map of the existing document.
     * @param sourceAndMetadataMap The metadata map of the new document.
     * @param processMap The current map being processed.
     * @return A filtered map containing only elements that require new embeddings.
     */
    public Map<String, Object> filterAndCopyExistingEmbeddings(
        Map<String, Object> existingSourceAndMetadataMap,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> processMap
    ) {
        return filterAndCopyExistingEmbeddings(existingSourceAndMetadataMap, sourceAndMetadataMap, processMap, "");
    }

    /**
     * Helper method for filter
     * @param existingSourceAndMetadataMap The metadata map of the existing document.
     * @param sourceAndMetadataMap The metadata map of the new document.
     * @param processMap The current map being processed.
     * @param traversedPath The dot-notation path of the previously traversed elements.
     * e.g:
     * In a map structured like:
     *       level1
     *         level2
     *            level3
     * traversedPath would be dot-separated string of level1.level2.level3
     * @return A filtered map containing only elements that require new embeddings
     */
    private Map<String, Object> filterAndCopyExistingEmbeddings(
        Map<String, Object> existingSourceAndMetadataMap,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> processMap,
        String traversedPath
    ) {
        Map<String, Object> filteredProcessMap = new HashMap<>();
        Map<String, Object> castedProcessMap = ProcessorUtils.unsafeCastToObjectMap(processMap);
        for (Map.Entry<String, Object> entry : castedProcessMap.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            String currentPath = traversedPath.isEmpty() ? key : traversedPath + "." + key;
            if (value instanceof Map) {
                Map<String, Object> filteredInnerMap = filterAndCopyExistingEmbeddings(
                    existingSourceAndMetadataMap,
                    sourceAndMetadataMap,
                    ProcessorUtils.unsafeCastToObjectMap(value),
                    currentPath
                );
                filteredProcessMap.put(key, filteredInnerMap.isEmpty() ? null : filteredInnerMap);
            } else if (value instanceof List) {
                List<Object> processedList = filterListValue(
                    currentPath,
                    ProcessorUtils.unsafeCastToObjectList(value),
                    sourceAndMetadataMap,
                    existingSourceAndMetadataMap
                );
                filteredProcessMap.put(key, processedList);
            } else if (value instanceof String) {
                Object processedValue = filterInferenceValue(currentPath, value, sourceAndMetadataMap, existingSourceAndMetadataMap, -1);
                filteredProcessMap.put(key, processedValue);
            }
        }
        return filteredProcessMap;
    }

    /**
     * Processes a list of values by comparing them against source and existing metadata.
     *
     * @param embeddingKey The current path in dot notation for the list being processed
     * @param processList The list of values to process
     * @param sourceAndMetadataMap The metadata map of the new document
     * @param existingSourceAndMetadataMap The metadata map of the existing document
     * @return A processed list containing non-filtered elements
     */
    protected List<Object> filterListValue(
        String embeddingKey,
        List<Object> processList,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap
    ) {
        String textKey = reversedFieldMap.get(embeddingKey);
        Optional<Object> existingListOptional = ProcessorUtils.getValueFromSource(existingSourceAndMetadataMap, textKey);
        Optional<Object> embeddingListOptional = ProcessorUtils.getValueFromSource(existingSourceAndMetadataMap, embeddingKey);
        if (existingListOptional.isPresent() == false || embeddingListOptional.isPresent() == false) {
            return processList;
        }
        List<Object> existingListValue = ProcessorUtils.unsafeCastToObjectList(existingListOptional.get());
        if (existingListValue.getFirst() instanceof List) {
            // in case of nested list, compare and copy by list comparison
            return copyEmbeddingForListObject(
                embeddingKey,
                processList,
                ProcessorUtils.unsafeCastToObjectList(existingListValue.getFirst()),
                ProcessorUtils.unsafeCastToObjectList(embeddingListOptional.get()),
                sourceAndMetadataMap
            );
        } else {
            // in case of List of Maps, compare each map entry in list
            return filterMapValuesInList(
                embeddingKey,
                processList,
                ProcessorUtils.unsafeCastToObjectList(existingListOptional.get()),
                ProcessorUtils.unsafeCastToObjectList(embeddingListOptional.get()),
                sourceAndMetadataMap
            );
        }
    }

    /**
     * Processes a list containing map values by iterating through each item and processing it individually.
     *
     * @param embeddingKey The current path in dot notation
     * @param processList The list of Map items to process
     * @param existingList The list of existing Map items form comparison
     * @param embeddingList The list of existing embeddings
     * @param sourceAndMetadataMap The metadata map of the new document
     * @return A processed list containing non-filtered elements
     */
    public List<Object> filterMapValuesInList(
        String embeddingKey,
        List<Object> processList,
        List<Object> existingList,
        List<Object> embeddingList,
        Map<String, Object> sourceAndMetadataMap
    ) {
        List<Object> filteredList = new ArrayList<>();
        ListIterator<Object> processListIterator = processList.listIterator();
        ListIterator<Object> existingListIterator = existingList.listIterator();
        ListIterator<Object> embeddingListIterator = embeddingList.listIterator();
        int index = 0;
        for (Object processValue : processList) {
            if (Objects.nonNull(processValue) && existingListIterator.hasNext() && embeddingListIterator.hasNext()) {
                Object processedItem = copyEmbeddingForSingleObject(
                    embeddingKey,
                    processValue,
                    existingListIterator.next(),
                    embeddingListIterator.next(),
                    sourceAndMetadataMap,
                    index
                );
                filteredList.add(processedItem);
            }
            index++;
        }
        return filteredList;
    }
}
