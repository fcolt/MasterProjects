import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.tika.exception.TikaException;

public class IndexSearcher {
    private final String indexDir;
    private final String dataDir;
    public int filesIndexed;

    public IndexSearcher (String dataDir, String indexDir) throws IOException, TikaException {
        this.dataDir = dataDir;
        this.indexDir = indexDir;
        createIndex();
    }

    private void createIndex() throws IOException, TikaException {
        FileUtils.cleanDirectory(new File(indexDir));

        DocumentIndexer indexer = new DocumentIndexer(indexDir);
        long startTime = System.currentTimeMillis();
        filesIndexed = indexer.createIndex(dataDir, new AcceptedFileTypes());
        long endTime = System.currentTimeMillis();
        indexer.close();
        System.out.println(filesIndexed + " files indexed in "
                + (endTime - startTime) * 0.001 + "s");
    }

    public void search(String searchQuery, int noOfHits) throws IOException, ParseException {
        Searcher searcher = new Searcher(indexDir);
        long startTime = System.currentTimeMillis();
        TopDocs hits = searcher.search(searchQuery, noOfHits);
        long endTime = System.currentTimeMillis();

        System.out.println(hits.totalHits + " document(s) found in " + (endTime - startTime) * 0.001 + "s" + " for query: " + searchQuery);
        for (ScoreDoc scoreDoc : hits.scoreDocs) {
            Document doc = searcher.getDocument(scoreDoc);
            System.out.println("Filepath: " + doc.get("filepath"));
        }
    }
}