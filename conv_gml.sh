#gml2gv best_cppn.gml > best_cppn.dot
for i in *_best_cppn.dot; do
        dot -Tpdf $i > $i.pdf
done
#evince best_cppn.pdf
