/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.optimization;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.get.GetAction;
import org.opensearch.action.get.GetRequest;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.action.ActionListener;
import org.opensearch.env.Environment;
import org.opensearch.ingest.IngestDocument;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.processor.TextInferenceRequest;
import org.opensearch.neuralsearch.processor.util.ProcessorUtils;
import org.opensearch.transport.client.OpenSearchClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * This processor is used for selective text embedding processing. This processor will skip redundant inference calls by comparing existing document and new document.
 * If inference texts stay the same, the SelectiveTextEmbeddingProcessor will copy over existing embeddings and mark null inference text from process map
 */
@Log4j2
public class SelectiveTextEmbeddingProcessor extends SelectiveInferenceProcessor {
    public static final String TYPE = "text_embedding";
    public static final String LIST_TYPE_NESTED_MAP_KEY = "knn";
    private static final String INDEX_FIELD = "_index";
    private static final String ID_FIELD = "_id";

    private final OpenSearchClient openSearchClient;

    public SelectiveTextEmbeddingProcessor(
        String tag,
        String description,
        int batchSize,
        String modelId,
        Map<String, Object> fieldMap,
        OpenSearchClient openSearchClient,
        MLCommonsClientAccessor clientAccessor,
        Environment environment,
        ClusterService clusterService
    ) {
        super(tag, description, batchSize, modelId, TYPE, LIST_TYPE_NESTED_MAP_KEY, fieldMap, clientAccessor, environment, clusterService);
        this.openSearchClient = openSearchClient;
    }

    @Override
    public void doExecute(
        IngestDocument ingestDocument,
        Map<String, Object> ProcessMap,
        List<String> inferenceList,
        BiConsumer<IngestDocument, Exception> handler
    ) {
        String index = ingestDocument.getSourceAndMetadata().get(INDEX_FIELD).toString();
        String id = ingestDocument.getSourceAndMetadata().get(ID_FIELD).toString();
        openSearchClient.execute(GetAction.INSTANCE, new GetRequest(index, id), ActionListener.wrap(response -> {
            final Map<String, Object> document = response.getSourceAsMap();
            if (document == null || document.isEmpty()) {
                makeInferenceCall(ingestDocument, ProcessMap, inferenceList, handler, false);
            } else {
                Map<String, Object> filteredProcessMap = filterProcessMap(document, ingestDocument.getSourceAndMetadata(), ProcessMap);
                List<String> filteredInferenceList = createInferenceList(filteredProcessMap).stream()
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
                if (!filteredInferenceList.isEmpty()) {
                    makeInferenceCall(ingestDocument, filteredProcessMap, filteredInferenceList, handler, true);
                } else {
                    handler.accept(ingestDocument, null);
                }
            }
        }, e -> { handler.accept(null, e); }));
    }

    @Override
    protected void doBatchExecute(List<String> inferenceList, Consumer<List<?>> handler, Consumer<Exception> onException) {
        mlCommonsClientAccessor.inferenceSentences(
            TextInferenceRequest.builder().modelId(this.modelId).inputTexts(inferenceList).build(),
            ActionListener.wrap(handler::accept, onException)
        );
    }

    /**
     * This method checks for the following requirements to determine whether embeddings can be copied from existingSourceAndMetadataMap to sourceAndMetadataMap
     *  - inference text is the same between existingSourceAndMetadataMap and sourceAndMetadataMap
     *  - existing existingSourceAndMetadataMap has embeddings for inference text
     * @param currentPath path to embedding field
     * @param processValue value to be checked for whether embeddings can be copied over
     * @param level current level in the process map traversal
     * @param sourceAndMetadataMap SourceAndMetadataMap of ingestDocument Document
     * @param existingSourceAndMetadataMap SourceAndMetadataMap of existing Document
     * @param index index of sourceValue if in list
     *
     * returns null if existing embedding was successfully populated after passing the required checks and filtered, return processValue otherwise
     */
    public Object processValue(
        String currentPath,
        Object processValue,
        int level,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap,
        int index
    ) {
        String textKey = ProcessorUtils.findKeyFromFromValue(fieldMap, currentPath, level);
        if (textKey == null) {
            return processValue;
        }

        String fullTextKey = ProcessorUtils.computeFullTextKey(currentPath, textKey, level);
        String fullEmbeddingKey = currentPath;

        Optional<Object> sourceValue = ProcessorUtils.getValueFromSource(sourceAndMetadataMap, fullTextKey, index);
        Optional<Object> existingValue = ProcessorUtils.getValueFromSource(existingSourceAndMetadataMap, fullTextKey, index);
        Optional<Object> embeddingValue = ProcessorUtils.getValueFromSource(existingSourceAndMetadataMap, fullEmbeddingKey, index);

        if (sourceValue.isPresent() && existingValue.isPresent() && sourceValue.get().equals(existingValue.get())) {
            embeddingValue.ifPresent(o -> ProcessorUtils.setValueToSource(sourceAndMetadataMap, fullEmbeddingKey, o, index));
            return null;
        }
        return processValue;
    }

    /**
     * This method checks for the following requirements to determine whether embeddings can be copied from existingSourceAndMetadataMap to sourceAndMetadataMap
     *  - In the given list of inference texts, inference texts in the same index is the same between existingSourceAndMetadataMap and sourceAndMetadataMap
     *  - existing existingSourceAndMetadataMap has embeddings for corresponding inference texts
     * @param processList list of inference texts to check for
     * @param sourceListOptional Optional list of inference texts in sourceAndMetadataMap
     * @param existingListOptional Optional list of inference texts in existingSourceAndMetadataMap
     * @param embeddingListOptional Optional list of embeddings for inference texts
     * @param sourceAndMetadataMap SourceAndMetadataMap of ingestDocument Document
     * @param existingSourceAndMetadataMap SourceAndMetadataMap of existing Document
     * @param fullEmbeddingKey path to embedding key
     *
     * returns filtered list with copied embeddings for equal inference texts. If embeddings could not be copied over, the corresponding index is marked 'null' to be populated later with embeddings retrieved from inference call.
     */
    @Override
    public List<Object> processValues(
        List<?> processList,
        Optional<Object> sourceListOptional,
        Optional<Object> existingListOptional,
        Optional<Object> embeddingListOptional,
        Map<String, Object> sourceAndMetadataMap,
        Map<String, Object> existingSourceAndMetadataMap,
        String fullEmbeddingKey
    ) {
        List<Object> filteredList = new ArrayList<>();
        List<Object> updatedEmbeddings = new ArrayList<>();
        if (sourceListOptional.isPresent() && existingListOptional.isPresent()) {
            if (sourceListOptional.get() instanceof List
                && existingListOptional.get() instanceof List
                && embeddingListOptional.get() instanceof List) {
                List sourceList = (ArrayList) sourceListOptional.get();
                List existingList = (ArrayList) existingListOptional.get();
                List embeddingList = (ArrayList) embeddingListOptional.get();
                int min = Math.min(sourceList.size(), existingList.size());
                for (int j = 0; j < min; j++) {
                    if (sourceList.get(j).equals(existingList.get(j))) {
                        updatedEmbeddings.add((embeddingList).get(j));
                        filteredList.add(null);
                    } else {
                        filteredList.add(processList.get(j));
                        updatedEmbeddings.add(null);
                    }
                }
                ProcessorUtils.setValueToSource(sourceAndMetadataMap, fullEmbeddingKey, updatedEmbeddings);
            }
        } else {
            return (List) processList;
        }
        return filteredList;
    }
}
