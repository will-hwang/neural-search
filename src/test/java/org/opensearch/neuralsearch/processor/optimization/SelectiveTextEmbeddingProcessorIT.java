/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.optimization;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.apache.lucene.search.join.ScoreMode;
import org.junit.Before;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.neuralsearch.BaseNeuralSearchIT;
import org.opensearch.neuralsearch.query.NeuralQueryBuilder;

public class SelectiveTextEmbeddingProcessorIT extends BaseNeuralSearchIT {

    private static final String INDEX_NAME = "selective_text_embedding_index";

    private static final String PIPELINE_NAME = "selective_pipeline";
    protected static final String QUERY_TEXT = "hello";
    protected static final String LEVEL_1_FIELD = "nested_passages";
    protected static final String LEVEL_2_FIELD = "level_2";
    protected static final String LEVEL_3_FIELD_TEXT = "level_3_text";
    protected static final String LEVEL_3_FIELD_CONTAINER = "level_3_container";
    protected static final String LEVEL_3_FIELD_EMBEDDING = "level_3_embedding";
    protected static final String TEXT_FIELD_VALUE_1 = "hello";
    protected static final String TEXT_FIELD_VALUE_2 = "joker";
    protected static final String TEXT_FIELD_VALUE_3 = "def";
    private final String INGEST_DOC1 = Files.readString(Path.of(classLoader.getResource("processor/ingest_doc1.json").toURI()));
    private final String INGEST_DOC3 = Files.readString(Path.of(classLoader.getResource("processor/ingest_doc3.json").toURI()));
    private final String INGEST_DOC4 = Files.readString(Path.of(classLoader.getResource("processor/ingest_doc4.json").toURI()));
    private final String INGEST_DOC5 = Files.readString(Path.of(classLoader.getResource("processor/ingest_doc5.json").toURI()));
    private final String UPDATE_DOC1 = Files.readString(Path.of(classLoader.getResource("processor/update_doc1.json").toURI()));
    private final String UPDATE_DOC3 = Files.readString(Path.of(classLoader.getResource("processor/update_doc3.json").toURI()));
    private final String UPDATE_DOC4 = Files.readString(Path.of(classLoader.getResource("processor/update_doc4.json").toURI()));
    private final String UPDATE_DOC5 = Files.readString(Path.of(classLoader.getResource("processor/update_doc5.json").toURI()));

    public SelectiveTextEmbeddingProcessorIT() throws IOException, URISyntaxException {}

    @Before
    public void setUp() throws Exception {
        super.setUp();
        updateClusterSettings();
    }

    public void testTextEmbeddingProcessorWithSkipExisting() throws Exception {
        String modelId = uploadTextEmbeddingModel();
        loadModel(modelId);
        createPipelineProcessor(modelId, PIPELINE_NAME, ProcessorType.TEXT_EMBEDDING_WITH_SKIP_EXISTING);
        createIndexWithPipeline(INDEX_NAME, "IndexMappings.json", PIPELINE_NAME);
        ingestDocument(INDEX_NAME, INGEST_DOC1, "1");
        updateDocument(INDEX_NAME, UPDATE_DOC1, "1");
        assertEquals(1, getDocCount(INDEX_NAME));
        assertEquals(2, getDocById(INDEX_NAME, "1").get("_version"));
    }

    public void testNestedFieldMapping_whenDocumentsIngestedAndUpdated_thenSuccessful() throws Exception {
        String modelId = uploadTextEmbeddingModel();
        loadModel(modelId);
        createPipelineProcessor(modelId, PIPELINE_NAME, ProcessorType.TEXT_EMBEDDING_WITH_NESTED_FIELDS_MAPPING_WITH_SKIP_EXISTING);
        createIndexWithPipeline(INDEX_NAME, "IndexMappings.json", PIPELINE_NAME);
        ingestDocument(INDEX_NAME, INGEST_DOC3, "3");
        updateDocument(INDEX_NAME, UPDATE_DOC3, "3");
        ingestDocument(INDEX_NAME, INGEST_DOC4, "4");
        updateDocument(INDEX_NAME, UPDATE_DOC4, "4");

        assertDoc((Map<String, Object>) getDocById(INDEX_NAME, "3").get("_source"), TEXT_FIELD_VALUE_1, Optional.of(TEXT_FIELD_VALUE_3));
        assertDoc((Map<String, Object>) getDocById(INDEX_NAME, "4").get("_source"), TEXT_FIELD_VALUE_2, Optional.empty());

        NeuralQueryBuilder neuralQueryBuilderQuery = NeuralQueryBuilder.builder()
            .fieldName(LEVEL_1_FIELD + "." + LEVEL_2_FIELD + "." + LEVEL_3_FIELD_CONTAINER + "." + LEVEL_3_FIELD_EMBEDDING)
            .queryText(QUERY_TEXT)
            .modelId(modelId)
            .k(10)
            .build();

        QueryBuilder queryNestedLowerLevel = QueryBuilders.nestedQuery(
            LEVEL_1_FIELD + "." + LEVEL_2_FIELD,
            neuralQueryBuilderQuery,
            ScoreMode.Total
        );
        QueryBuilder queryNestedHighLevel = QueryBuilders.nestedQuery(LEVEL_1_FIELD, queryNestedLowerLevel, ScoreMode.Total);

        Map<String, Object> searchResponseAsMap = search(INDEX_NAME, queryNestedHighLevel, 2);
        assertNotNull(searchResponseAsMap);

        Map<String, Object> hits = (Map<String, Object>) searchResponseAsMap.get("hits");
        assertNotNull(hits);

        assertEquals(1.0, hits.get("max_score"));
        List<Map<String, Object>> listOfHits = (List<Map<String, Object>>) hits.get("hits");
        assertNotNull(listOfHits);
        assertEquals(2, listOfHits.size());

        Map<String, Object> innerHitDetails = listOfHits.get(0);
        assertEquals("3", innerHitDetails.get("_id"));
        assertEquals(1.0, innerHitDetails.get("_score"));

        innerHitDetails = listOfHits.get(1);
        assertEquals("4", innerHitDetails.get("_id"));
        assertTrue((double) innerHitDetails.get("_score") <= 1.0);
    }

    public void testNestedFieldMapping_whenDocumentInListIngestedAndUpdated_thenSuccessful() throws Exception {
        String modelId = uploadTextEmbeddingModel();
        loadModel(modelId);
        createPipelineProcessor(modelId, PIPELINE_NAME, ProcessorType.TEXT_EMBEDDING_WITH_NESTED_FIELDS_MAPPING);
        createIndexWithPipeline(INDEX_NAME, "IndexMappings.json", PIPELINE_NAME);
        ingestDocument(INDEX_NAME, INGEST_DOC5, "5");
        updateDocument(INDEX_NAME, UPDATE_DOC5, "5");

        assertDocWithLevel2AsList((Map<String, Object>) getDocById(INDEX_NAME, "5").get("_source"));

        NeuralQueryBuilder neuralQueryBuilderQuery = NeuralQueryBuilder.builder()
            .fieldName(LEVEL_1_FIELD + "." + LEVEL_2_FIELD + "." + LEVEL_3_FIELD_CONTAINER + "." + LEVEL_3_FIELD_EMBEDDING)
            .queryText(QUERY_TEXT)
            .modelId(modelId)
            .k(10)
            .build();

        QueryBuilder queryNestedLowerLevel = QueryBuilders.nestedQuery(
            LEVEL_1_FIELD + "." + LEVEL_2_FIELD,
            neuralQueryBuilderQuery,
            ScoreMode.Total
        );
        QueryBuilder queryNestedHighLevel = QueryBuilders.nestedQuery(LEVEL_1_FIELD, queryNestedLowerLevel, ScoreMode.Total);

        Map<String, Object> searchResponseAsMap = search(INDEX_NAME, queryNestedHighLevel, 2);
        assertNotNull(searchResponseAsMap);

        assertEquals(1, getHitCount(searchResponseAsMap));

        Map<String, Object> innerHitDetails = getFirstInnerHit(searchResponseAsMap);
        assertEquals("5", innerHitDetails.get("_id"));
    }

    private String uploadTextEmbeddingModel() throws Exception {
        String requestBody = Files.readString(Path.of(classLoader.getResource("processor/UploadModelRequestBody.json").toURI()));
        return registerModelGroupAndUploadModel(requestBody);
    }

    private void assertDoc(Map<String, Object> sourceMap, String textFieldValue, Optional<String> level3ExpectedValue) {
        assertNotNull(sourceMap);
        assertTrue(sourceMap.containsKey(LEVEL_1_FIELD));
        Map<String, Object> nestedPassages = (Map<String, Object>) sourceMap.get(LEVEL_1_FIELD);
        assertTrue(nestedPassages.containsKey(LEVEL_2_FIELD));
        Map<String, Object> level2 = (Map<String, Object>) nestedPassages.get(LEVEL_2_FIELD);
        assertEquals(textFieldValue, level2.get(LEVEL_3_FIELD_TEXT));
        Map<String, Object> level3 = (Map<String, Object>) level2.get(LEVEL_3_FIELD_CONTAINER);
        List<Double> embeddings = (List<Double>) level3.get(LEVEL_3_FIELD_EMBEDDING);
        assertEquals(768, embeddings.size());
        for (Double embedding : embeddings) {
            assertTrue(embedding >= 0.0 && embedding <= 1.0);
        }
        if (level3ExpectedValue.isPresent()) {
            assertEquals(level3ExpectedValue.get(), level3.get("level_4_text_field"));
        }
    }

    private void assertDocWithLevel2AsList(Map<String, Object> sourceMap) {
        assertNotNull(sourceMap);
        assertTrue(sourceMap.containsKey(LEVEL_1_FIELD));
        assertTrue(sourceMap.get(LEVEL_1_FIELD) instanceof List);
        List<Map<String, Object>> nestedPassages = (List<Map<String, Object>>) sourceMap.get(LEVEL_1_FIELD);
        nestedPassages.forEach(nestedPassage -> {
            assertTrue(nestedPassage.containsKey(LEVEL_2_FIELD));
            Map<String, Object> level2 = (Map<String, Object>) nestedPassage.get(LEVEL_2_FIELD);
            Map<String, Object> level3 = (Map<String, Object>) level2.get(LEVEL_3_FIELD_CONTAINER);
            List<Double> embeddings = (List<Double>) level3.get(LEVEL_3_FIELD_EMBEDDING);
            assertEquals(768, embeddings.size());
            for (Double embedding : embeddings) {
                assertTrue(embedding >= 0.0 && embedding <= 1.0);
            }
        });
    }
}
