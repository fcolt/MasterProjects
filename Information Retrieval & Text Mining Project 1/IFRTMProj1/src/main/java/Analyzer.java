import java.io.File;
import java.io.IOException;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

public class Analyzer {
    private String indexDir = "src/main/resources/indices";
    private String dataDir = "src/main/resources/docs";
    private DocumentIndexer indexer;
    private Searcher searcher;

    public Analyzer(String indexDir, String dataDir) throws IOException {
        this.indexDir = indexDir;
        this.dataDir = dataDir;
        createIndex();
    }

    public void createIndex() throws IOException {

        //stergem indecsii deja creati
        File file = new File(indexDir);
        String[] myFiles;
        if(file.isDirectory()){
            myFiles = file.list();
            for (String s : myFiles) {
                File myFile = new File(file, s);
                myFile.delete();
            }
        }

        indexer = new DocumentIndexer(indexDir);
        int numIndexed;
        long startTime = System.currentTimeMillis();
        numIndexed = indexer.createIndex(dataDir, new AcceptedFileTypes());
        long endTime = System.currentTimeMillis();
        indexer.close();
        System.out.println(numIndexed+" fisiere indexate in: "
                +(endTime-startTime)+" ms");
    }

    public void search(String searchQuery) throws IOException, ParseException {
        searcher = new Searcher(indexDir);
        long startTime = System.currentTimeMillis();
        TopDocs hits = searcher.search(searchQuery);
        long endTime = System.currentTimeMillis();

        System.out.println(hits.totalHits +
                (hits.totalHits.value == 1 ? " document gasit. Timp :" : " documente gasite. Timp:")
                + (endTime - startTime));
        for(ScoreDoc scoreDoc : hits.scoreDocs) {
            Document doc = searcher.getDocument(scoreDoc);
            System.out.println("Fisier: " + doc.get("filepath"));
        }
    }
}