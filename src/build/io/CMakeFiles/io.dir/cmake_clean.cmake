file(REMOVE_RECURSE
  "libio.a"
  "libio.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/io.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
