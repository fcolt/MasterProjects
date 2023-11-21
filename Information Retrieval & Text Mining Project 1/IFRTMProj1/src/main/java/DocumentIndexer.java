import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Objects;

public class DocumentIndexer {

    private IndexWriter writer;

    public DocumentIndexer(String indexDirectoryPath) throws IOException {
        Directory indexDirectory = FSDirectory.open(Paths.get(indexDirectoryPath));

        BetterRomanianAnalyzer analyzer = new BetterRomanianAnalyzer();

        IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        writer = new IndexWriter(indexDirectory, iwc);
    }

    public void close() throws IOException {
        writer.close();
    }

    private Document getDocument(File file) throws IOException, TikaException {
        Tika tika = new Tika();
        String filetype = tika.detect(file);
        System.out.println("Indexed:" + filetype);

        Document document = new Document();

        TextField contentField = new TextField("contents", tika.parseToString(file), TextField.Store.YES);

        TextField fileNameField = new TextField("filename",
                file.getName(), TextField.Store.YES);
        TextField filePathField = new TextField("filepath",
                file.getCanonicalPath(), TextField.Store.YES);

        document.add(contentField);
        document.add(fileNameField);
        document.add(filePathField);

        return document;
    }

    private void indexFile(File file) throws IOException, TikaException {
        System.out.println("Indexing " + file.getCanonicalPath());
        Document document = getDocument(file);
        writer.addDocument(document);
    }

    public int createIndex(String dataDirPath, FileFilter filter) throws IOException, TikaException {
        File[] files = new File(dataDirPath).listFiles();

        if (Objects.isNull(files)) {
            return 0;
        }

        for (File file : files) {
            if (!file.isDirectory()
                    && !file.isHidden()
                    && file.exists()
                    && file.canRead()
                    && filter.accept(file)
            ) {
                indexFile(file);
            }
        }

        return writer.getDocStats().numDocs;
    }
}