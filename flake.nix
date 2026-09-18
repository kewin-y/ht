{
  description = "makemor";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    forAllSystems = nixpkgs.lib.genAttrs [
      "aarch64-linux"
      "x86_64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];
  in {
    devShells = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
      in {
        default = pkgs.mkShell {
          buildInputs = lib.attrValues {
            inherit
              (pkgs)
              ty
              ruff
              uv
              ;
          };
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [stdenv.cc.cc.lib zlib glib])}:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
  };
}
