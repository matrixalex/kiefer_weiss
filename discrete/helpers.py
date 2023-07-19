from typing import Type

from config import DistributionTypeEnum
from discrete.utils.binom import BinomDistributionHandler
from discrete.utils.core import BaseDistributionHandler
from discrete.utils.nbinom import NBinomDistributionHandler
from discrete.utils.pois import PoissonDistributionHandler


class DistributionController:

    @staticmethod
    def get_distribution_handler(distribution_type) -> Type[BaseDistributionHandler]:
        distribution_map = {
            DistributionTypeEnum.BINOM: BinomDistributionHandler,
            DistributionTypeEnum.NBINOM: NBinomDistributionHandler,
            DistributionTypeEnum.POISSON: PoissonDistributionHandler
        }

        return distribution_map[distribution_type]
