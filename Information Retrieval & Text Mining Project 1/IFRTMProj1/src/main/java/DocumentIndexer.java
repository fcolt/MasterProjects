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

public class DocumentIndexer {

    private IndexWriter writer;

    public DocumentIndexer(String indexDirectoryPath) throws IOException {
        Directory indexDirectory =
                FSDirectory.open(Paths.get(indexDirectoryPath));

        RomanianAnalyzer analyzer = new RomanianAnalyzer();

        IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        writer = new IndexWriter(indexDirectory, iwc);
    }

    public void close() throws IOException {
        writer.close();
    }

    private Document getDocument(File file) throws IOException {
        Tika tika = new Tika();
        String filetype = tika.detect(file);
        System.out.println("indexat:" + filetype);

        //se creaza un nou document
        //si se adauga 3 field-uri:
        //content, filename si filepath
        Document document = new Document();

        TextField contentField = null;
        try {
            contentField = new TextField("contents", tika.parseToString(file), TextField.Store.YES);
        } catch (TikaException e) {
            e.printStackTrace();
        }

        TextField fileNameField = new TextField("filename",
                file.getName(), TextField.Store.YES);
        TextField filePathField = new TextField("filepath",
                file.getCanonicalPath(), TextField.Store.YES);

        document.add(contentField);
        document.add(fileNameField);
        document.add(filePathField);

        return document;
    }

    private void indexFile(File file) throws IOException {
        System.out.println("Se indexeaza " + file.getCanonicalPath());
        Document document = getDocument(file);
        writer.addDocument(document);
    }

    public int createIndex(String dataDirPath, FileFilter filter) throws IOException {
        File[] files = new File(dataDirPath).listFiles();

        if (files != null) {
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
        }
        return writer.getDocStats().numDocs;
    }
}